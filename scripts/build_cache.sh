#!/bin/bash
# boltz-neuron: compile and package the caches for the inputs you will serve.
#
# Run on a large-RAM inf2 host (inf2.8xlarge) AFTER scripts/setup_inf2.sh. neuronx-cc needs about
# 40 GB of host RAM per graph, so an inf2.xlarge (~15 GB) cannot compile -- but it can run from a
# cache built here (same Inferentia2 chip). Restore the result with scripts/restore_cache.sh.
#
# Usage:  bash scripts/build_cache.sh [INPUT] [DS_LIST]
#   INPUT    a Boltz input YAML, or a directory of them (default: examples/8eil.yaml).
#            Each distinct (padded token count, atom count, diffusion_samples) is its own
#            compiled shape, so include one input of every size you will serve.
#   DS_LIST  comma-separated --diffusion_samples values (default: 1,8,25)
#
# Build ALL shapes in ONE run: NEFF and AOTAutograd entries from several archives merge when
# restored, but the dynamo precompile package is a single file per function, so a later archive
# replaces an earlier one's package.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${BOLTZ_NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
INPUT="${1:-$REPO/examples/8eil.yaml}"
DS_LIST="${2:-1,8,25}"
CACHE="/tmp/neff_cache"
OUT="$HOME/boltz_neuron_cache.tgz"
ARGS="--recycling_steps 3 --sampling_steps 50 --num_workers 0 --override"

source "$VENV/bin/activate"
export NEURON_RT_VISIBLE_CORES=0

echo "=== compiling $INPUT for diffusion_samples in {$DS_LIST} (first run per shape takes minutes) ==="
for ds in ${DS_LIST//,/ }; do
  python -m boltz_neuron predict "$INPUT" --out_dir "/tmp/warm_ds$ds" $ARGS --diffusion_samples "$ds" \
    > "/tmp/warm_ds$ds.log" 2>&1 && echo "  ds=$ds compiled OK" || { echo "  ds=$ds FAILED"; tail -5 "/tmp/warm_ds$ds.log"; }
done

# Dynamo precompile package (BOLTZ_NEURON_PRECOMPILE=1): does not load on torch 2.14 / SDK 2.33 (guard
# rebuild fails with "IndexError: list index out of range"), so it is only recorded on request.
if [ "${BOLTZ_NEURON_BUILD_PRECOMPILE:-0}" = 1 ]; then
  echo "=== recording the dynamo precompile package ==="
  for ds in ${DS_LIST//,/ }; do
    BOLTZ_NEURON_PRECOMPILE=1 python -m boltz_neuron predict "$INPUT" --out_dir "/tmp/warm_pc_ds$ds" $ARGS --diffusion_samples "$ds" \
      > "/tmp/warm_pc_ds$ds.log" 2>&1 && echo "  ds=$ds precompile OK" || { echo "  ds=$ds precompile FAILED"; tail -5 "/tmp/warm_pc_ds$ds.log"; }
  done
fi

echo "=== packaging -> $OUT ==="
if [ ! -d "$CACHE" ]; then
  echo "ERROR: $CACHE not found -- did any predict run succeed?"; exit 1
fi
IND="/tmp/torchinductor_$(id -un)"
# AOTAutograd entries live in $IND/aotautograd; with precompile, torch_neuronx uses $IND/rank_0.
rm -rf /tmp/aot_bundle; mkdir -p /tmp/aot_bundle; EXTRA=""
[ -d "$IND/aotautograd" ] && { cp -r "$IND/aotautograd" /tmp/aot_bundle/; EXTRA="aot_bundle"; }
[ -d "$IND/rank_0" ] && { cp -r "$IND/rank_0" /tmp/aot_bundle/; EXTRA="aot_bundle"; }
tar czf "$OUT" -C "$(dirname "$CACHE")" "$(basename "$CACHE")" $EXTRA
rm -rf /tmp/aot_bundle
echo "packaged $(du -h "$OUT" | cut -f1). On the target instance, after setup_inf2.sh:"
echo "  bash scripts/restore_cache.sh $(basename "$OUT")"
