#!/bin/bash
# boltz-neuron setup for AWS Inferentia2 (inf2) on the public Neuron SDK 2.33 DLAMI.
#
# Host: "Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 24.04) SDK 2.33.0" (or the 2.33
# multi-framework DLAMI, which has the same venv). Installs Boltz-2 v2.2.1 into the DLAMI's
# torch-neuronx venv, applies two torch_neuronx fixes for Inferentia2 (NOTES.md) and installs the
# boltz_neuron package. Takes about 2 minutes. Safe to re-run.
set -e

VENV="${BOLTZ_NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (has boltz_neuron/)

echo "=== [1/5] Neuron device + venv check ==="
neuron-ls | head -8 || { echo "neuron-ls failed -- not a Neuron instance?"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "$VENV not found -- use the SDK 2.33 torch-neuronx DLAMI"; exit 1; }
source "$VENV/bin/activate"
python -c "import torch_neuronx" 2>/dev/null || { echo "torch_neuronx not importable in $VENV"; exit 1; }

echo "=== [2/5] 64 GB swap (neuronx-cc host-RAM protection) ==="
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap --force /swapfile >/dev/null && sudo swapon /swapfile
fi
swapon --show

echo "=== [3/5] Boltz-2 v2.2.1 + dependencies (into $VENV) ==="
# --no-deps keeps the DLAMI's torch / numpy / torch-neuronx untouched. Boltz 2.2.1 pins numpy<2,
# but runs on the DLAMI's numpy 2.5 (validated).
pip install -q uv
uv pip install -q --no-deps boltz==2.2.1 \
  einx==0.3.0 mashumaro==3.14 numba llvmlite omegaconf "antlr4-python3-runtime==4.9.3" \
  pytorch-lightning==2.6.6 lightning-utilities torchmetrics fairscale==0.4.13 rdkit modelcif==1.2 ihm \
  biopython==1.84 gemmi==0.6.5 chembl_structure_pipeline==1.2.2 dm-tree==0.1.8 hydra-core==1.3.2 \
  frozendict typing_inspect mypy_extensions click==8.1.7 types-requests wrapt msgpack
python -c "import boltz.main; print('  boltz OK')" 2>&1 | grep -v "RNG"

echo "=== [4/5] torch_neuronx fixes for Inferentia2 ==="
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
echo "  source $VENV/bin/activate"
echo "  export NEURON_RT_VISIBLE_CORES=0"
echo "  python -m boltz_neuron predict input.yaml --out_dir out --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0"
echo "Restore a prebuilt cache first (see README) or the first run of each shape compiles."
