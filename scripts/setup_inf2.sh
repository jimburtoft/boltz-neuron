#!/bin/bash
# boltz-neuron setup for AWS Inferentia2 (inf2).
#
# Installs the PyTorch Native Beta 5 stack + Boltz-2 v2.2.1 into ~/boltz_neuron_ws/workspace/venv,
# applies the Inferentia2 fixes listed in NOTES.md, and installs the boltz_neuron package.
# Takes about 20 minutes.
#
# Tested host: AWS Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.32 (20260818) on inf2.xlarge and
# inf2.8xlarge.
# Prereq (run from a machine with AWS credentials BEFORE this script, so the instance can pull
# the container):
#   TOKEN=$(aws ecr get-login-password --region us-east-1)
#   ssh <instance> "echo '$TOKEN' | sudo docker login --username AWS --password-stdin \
#        421672808698.dkr.ecr.us-east-1.amazonaws.com"
set -e

REPO="421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b"
# PyTorch Native Beta 5 (SDK 2.32-era): torch 2.12.1, torch-neuronx 2.12.3, neuronx-cc 2.27.2878, nki 0.6.0
BETA_DIGEST="${BOLTZ_BETA_DIGEST:-sha256:94413ce1ffea3d45757fa8b65dafac6b302a613e82b1e49fd99d3c70e1dffa8b}"
WS="$HOME/boltz_neuron_ws"
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (has boltz_neuron/)

echo "=== [1/6] Neuron device check ==="
neuron-ls || { echo "neuron-ls failed -- not a Neuron instance?"; exit 1; }

echo "=== [2/6] 64 GB swap (neuronx-cc host-RAM protection) ==="
if ! swapon --show | grep -q /swapfile; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
  sudo chmod 600 /swapfile; sudo mkswap --force /swapfile; sudo swapon /swapfile
fi

echo "=== [3/6] Pull Beta 5 container + extract /workspace ==="
sudo docker pull "${REPO}@${BETA_DIGEST}"
IMG=$(sudo docker images -q --filter reference=$REPO | head -1)
if [ ! -d "$WS/workspace" ]; then
  mkdir -p "$WS"
  CID=$(sudo docker create $IMG)
  sudo docker cp "$CID:/workspace" "$WS/workspace"
  sudo docker rm "$CID"
  sudo chown -R "$(id -u):$(id -g)" "$WS/workspace"
fi

echo "=== [4/6] Host runtime debs (match Beta 5) ==="
sudo dpkg -i "$WS"/workspace/runtime_artifacts/*.deb 2>/dev/null || \
  sudo dpkg -i --force-overwrite "$WS"/workspace/runtime_artifacts/*.deb || true

echo "=== [5/6] venv + wheels + Boltz-2 ==="
cd "$WS/workspace"
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip uv >/dev/null
export UV_PROJECT_ENVIRONMENT="$WS/workspace/venv"
uv pip install "$WS"/workspace/nki_wheels/nki-*-cp312-cp312-linux_x86_64.whl
uv pip install "$WS"/workspace/neuronx_cc_wheels/neuronx_cc-2.*-cp312-cp312-linux_x86_64.whl
( cd "$WS/workspace/torch_neuron_eager" && uv pip install -e ".[dev]" >/dev/null )
uv pip install --no-deps cloudpickle boltz==2.2.1 \
  einops einx mashumaro numba omegaconf pytorch-lightning fairscale rdkit modelcif ihm \
  biopython gemmi scikit-learn scipy chembl_structure_pipeline dm-tree torchmetrics hydra-core \
  "numpy<2.5" pandas python-dateutil six joblib narwhals threadpoolctl typer typing_inspect \
  "antlr4-python3-runtime==4.9.3" cuequivariance==0.8.1 pytz tzdata markupsafe fsspec \
  click frozendict llvmlite lightning-utilities msgpack aiohttp aiosignal frozenlist multidict \
  propcache yarl gitpython sentry-sdk protobuf pyyaml requests wandb docker-pycreds setproctitle \
  gitdb smmap aiohappyeyeballs >/dev/null 2>&1 || echo "  (some optional deps skipped)"
python -c "import boltz; print('boltz', 'OK')"

echo "=== [6/6] Inferentia2 fixes + install boltz_neuron package ==="
RS="$WS/workspace/torch_neuron_eager/torch_neuronx/python_ops/nki_kernels/rng_state.py"
if [ -f "$RS" ] && ! grep -q GEN2_SHIM "$RS"; then
  python3 - "$RS" << 'PY'
import re,sys
p=sys.argv[1]; s=open(p).read(); c="        set_rng_state_gpsimd(seed)\n"
if c in s:
    s=s.replace(c,"        # GEN2_SHIM\n        import os as _o\n        if any(g in _o.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE','').lower() for g in ('inf2','trn1','gen2')):\n            return None\n"+c,1)
    open(p,'w').write(s); print('  gen2 RNG shim applied')
PY
fi
# make boltz_neuron importable from the venv
cp -r "$PKG_DIR/boltz_neuron" "$WS/workspace/venv/lib/python3.12/site-packages/"

# nonzero decomposition dtype fix (NOTES.md): nonzero_with_count hard-codes fp32 scatter
# targets but scatter_adds model-dtype (bf16) values -> "scatter(): Expected self.dtype ...".
# Match the scatter targets to the values' dtype.
DEC="$WS/workspace/torch_neuron_eager/torch_neuronx/neuron_dynamo_backend/decompositions.py"
if [ -f "$DEC" ] && ! grep -q NONZERO_DTYPE_SHIM "$DEC"; then
  python3 - "$DEC" << 'PY'
import sys
p = sys.argv[1]; s = open(p).read()
s2 = s.replace("out_col = torch.zeros(size, dtype=torch.float, device=device)",
               "out_col = torch.zeros(size, dtype=col_vals.dtype, device=device)  # NONZERO_DTYPE_SHIM")
s2 = s2.replace("out_row = torch.zeros(size, dtype=torch.float, device=device)",
                "out_row = torch.zeros(size, dtype=row_vals.dtype, device=device)  # NONZERO_DTYPE_SHIM")
if s2 != s:
    open(p, "w").write(s2); print("  nonzero dtype shim applied")
else:
    print("  nonzero dtype shim: call sites not found (check decompositions.py version)")
PY
fi

# NOTE: the NEFF cache key hashes torch_neuronx source files (comments included). The lines these
# fixes insert must stay byte-identical to the ones the published HF cache was built with, or every
# graph recompiles on first run.
# nonzero COUNT fix (NOTES.md): on inf2/trn1, an integer sum of a bool tensor compiled in the same graph
# as a float reduction of it is silently miscounted (e.g. 1184 instead of 1568). nonzero_with_count
# hits this via `num_nonzero = mask.sum()`, so torch.nonzero / x[bool_mask] return too few rows and
# Boltz-2's structure writer crashes ("could not broadcast ... (1156,3) into (1540,3)") at some sizes
# (e.g. a 208-residue protein). Compute the count as a float sum instead (exact up to 2^24).
if [ -f "$DEC" ] && ! grep -q BOOL_SUM_SHIM "$DEC"; then
  python3 - "$DEC" << 'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    mask = tensor != 0\n    num_nonzero = mask.sum()\n"
new = ("    mask = tensor != 0\n"
       "    num_nonzero = mask.float().sum().to(torch.int64)  # BOOL_SUM_SHIM: int reduction of a bool co-scheduled with float reductions is miscounted on gen2\n")
if old in s:
    open(p, "w").write(s.replace(old, new, 1)); print("  nonzero count (bool-sum) shim applied")
else:
    print("  nonzero count shim: call site not found (check decompositions.py version)")
PY
fi

echo ""
echo "=== DONE ==="
echo "Activate and predict:"
echo "  source $WS/workspace/venv/bin/activate"
echo "  export NEURON_PLATFORM_TARGET_OVERRIDE=inf2 NEURON_RT_VISIBLE_CORES=0 NEURON_SCRATCHPAD_PAGE_SIZE=2048"
echo "  python -m boltz_neuron predict input.yaml --out_dir out --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0"
echo "Restore a prebuilt cache first (see README) or the first run of each shape compiles."
