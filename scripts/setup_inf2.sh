#!/bin/bash
# boltz-neuron setup for AWS Inferentia2 (inf2) and Trainium2 (trn2) on the public Neuron SDK 2.33 DLAMI.
#
# Host: "Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 24.04) SDK 2.33.0" (or the 2.33
# multi-framework DLAMI, which has the same venv). Installs Boltz-2 v2.2.1 into the DLAMI's
# torch-neuronx venv, applies two torch_neuronx fixes for Inferentia2 (NOTES.md) and installs the
# boltz_neuron package. Takes about 2 minutes. Safe to re-run. (Named setup_inf2.sh for history; the
# same script sets up trn2.)
set -e

VENV="${BOLTZ_NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (has boltz_neuron/)

echo "=== [1/5] Neuron device + venv check ==="
neuron-ls 2>/dev/null | head -8 || echo "  (neuron-ls unavailable here; continuing)"
if [ "$VENV" = system ]; then
  # Container images (e.g. public.ecr.aws/neuron/torch-neuronx:2.14.0-...-sdk2.33.0) have no venv.
  echo "  using the current python: $(command -v python3)"
  python() { python3 "$@"; }; pip() { python3 -m pip "$@"; }; export -f python pip
else
  [ -f "$VENV/bin/activate" ] || { echo "$VENV not found -- use the SDK 2.33 torch-neuronx DLAMI"; exit 1; }
  source "$VENV/bin/activate"
fi
python -c "import torch_neuronx" 2>/dev/null || { echo "torch_neuronx not importable in $VENV"; exit 1; }

echo "=== [2/5] 64 GB swap (neuronx-cc host-RAM protection) ==="
if [ "${BOLTZ_NEURON_SKIP_SWAP:-0}" = 1 ] || [ "$VENV" = system ]; then
  echo "  skipped"
elif ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap --force /swapfile >/dev/null && sudo swapon /swapfile
fi
swapon --show 2>/dev/null || true

echo "=== [3/5] Boltz-2 v2.2.1 + dependencies (into $VENV) ==="
# --no-deps keeps the DLAMI's torch / numpy / torch-neuronx untouched. Boltz 2.2.1 pins numpy<2,
# but runs on the DLAMI's numpy 2.5 (validated).
pip install -q uv
UVSYS=""; [ "$VENV" = system ] && UVSYS="--system --break-system-packages"
# Pins validated on the DLAMI venv (python 3.12). Python >= 3.13 (container images) has no wheels for
# the older gemmi / biopython / dm-tree, so there the newest wheels are used.
if python3 -c "import sys; sys.exit(sys.version_info < (3, 13))"; then
  OLD="biopython gemmi dm-tree mashumaro"
  # container images ship fewer packages than the DLAMI venv
  # Install the third-party dependencies WITH their own dependencies (pinning the image's torch/numpy),
  # then boltz itself with --no-deps (its torch/numpy pins would fight the image's).
  python3 -m pip freeze 2>/dev/null | grep -i -E "^(torch|torch-neuronx|numpy|neuronx-cc|nki|libneuronxla)==" > /tmp/bn_constraints.txt
  python3 -m uv pip install -q $UVSYS -c /tmp/bn_constraints.txt scikit-learn scipy pandas einops numba llvmlite \
    omegaconf "antlr4-python3-runtime==4.9.3" pytorch-lightning==2.6.6 torchmetrics fairscale==0.4.13 rdkit \
    modelcif==1.2 ihm biopython gemmi dm-tree mashumaro chembl_structure_pipeline==1.2.2 hydra-core==1.3.2 \
    einx==0.3.0 click==8.1.7 requests pyyaml tqdm
else
  OLD="biopython==1.84 gemmi==0.6.5 dm-tree==0.1.8 mashumaro==3.14"
fi
python3 -m uv pip install -q $UVSYS --no-deps boltz==2.2.1 \
  einx==0.3.0 numba llvmlite omegaconf "antlr4-python3-runtime==4.9.3" \
  pytorch-lightning==2.6.6 lightning-utilities torchmetrics fairscale==0.4.13 rdkit modelcif==1.2 ihm \
  $OLD chembl_structure_pipeline==1.2.2 hydra-core==1.3.2 \
  frozendict typing_inspect mypy_extensions click==8.1.7 types-requests wrapt msgpack
python -c "import boltz.main; print('  boltz OK')" 2>&1 | grep -v "RNG"

echo "=== [4/5] torch_neuronx fixes for Inferentia2 (harmless on trn2) ==="
# Both are in torch.nonzero's lowering (nonzero_with_count). They must be file edits: the compiled
# graph is built from this source. The NEFF cache key hashes torch_neuronx's source, so these lines
# must stay byte-identical to the ones the published cache was built with.
#  NONZERO_DTYPE_SHIM: fp32 scatter targets receive bf16 values -> "scatter(): Expected self.dtype".
#  BOOL_SUM_SHIM: an integer sum of a bool tensor is miscounted on Inferentia2 (e.g. 1060 instead of
#    1412), so torch.nonzero / x[bool_mask] return too few rows. Count with an fp32 sum instead.
DEC=$(python -c "import torch_neuronx.neuron_dynamo_backend.decompositions as d; print(d.__file__)" 2>/dev/null)
python3 - "$DEC" << 'PY'
import sys
p = sys.argv[1]; s = open(p).read(); s0 = s
s = s.replace("out_col = torch.zeros(size, dtype=torch.float, device=device)",
              "out_col = torch.zeros(size, dtype=col_vals.dtype, device=device)  # NONZERO_DTYPE_SHIM")
s = s.replace("out_row = torch.zeros(size, dtype=torch.float, device=device)",
              "out_row = torch.zeros(size, dtype=row_vals.dtype, device=device)  # NONZERO_DTYPE_SHIM")
s = s.replace("    mask = tensor != 0\n    num_nonzero = mask.sum()\n",
              "    mask = tensor != 0\n    num_nonzero = mask.to(torch.float32).sum().to(torch.int64)  # BOOL_SUM_SHIM\n", 1)
if s != s0:
    open(p, "w").write(s)
n = s.count("NONZERO_DTYPE_SHIM"), s.count("BOOL_SUM_SHIM")
print(f"  {p}: NONZERO_DTYPE_SHIM x{n[0]}, BOOL_SUM_SHIM x{n[1]}" + ("" if n == (2, 1) else "  <-- EXPECTED 2 and 1"))
PY

echo "=== [5/5] boltz_neuron package ==="
SP=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SP/boltz_neuron"
cp -r "$PKG_DIR/boltz_neuron" "$SP/"
python -c "import boltz_neuron; print('  boltz_neuron OK')"

echo ""
echo "=== DONE ==="
echo "Predict:"
[ "$VENV" = system ] || echo "  source $VENV/bin/activate"
echo "  export NEURON_RT_VISIBLE_CORES=0"
echo "  python -m boltz_neuron predict input.yaml --out_dir out --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0"
echo "Restore a prebuilt cache first (see README) or the first run of each shape compiles."
