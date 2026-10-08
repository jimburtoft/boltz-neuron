#!/bin/bash
# boltz-neuron: install caches built by scripts/build_cache.sh (or the prebuilt archive from
# https://huggingface.co/datasets/jburtoft/boltz-neuron-cache) so predict runs without compiling.
# Run AFTER scripts/setup_inf2.sh.
#
# Usage:  bash scripts/restore_cache.sh <cache.tgz>
set -eu
TGZ="${1:?usage: restore_cache.sh <cache.tgz>}"
tar xzf "$TGZ" -C /tmp
echo "restored NEFF cache -> /tmp/neff_cache ($(du -sh /tmp/neff_cache 2>/dev/null | cut -f1))"
if [ -d /tmp/aot_bundle ]; then
  IND="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$(id -un)}"
  mkdir -p "$IND"
  [ -d /tmp/aot_bundle/aotautograd ] && { cp -r /tmp/aot_bundle/aotautograd "$IND/"; echo "restored AOTAutograd cache -> $IND/aotautograd"; }
  [ -d /tmp/aot_bundle/rank_0 ] && { cp -r /tmp/aot_bundle/rank_0 "$IND/"; echo "restored precompile package -> $IND/rank_0 (used only with BOLTZ_NEURON_PRECOMPILE=1)"; }
  rm -rf /tmp/aot_bundle
fi
echo ""
echo "Predict:"
echo "  source \$HOME/boltz_neuron_ws/workspace/venv/bin/activate"
echo "  export NEURON_PLATFORM_TARGET_OVERRIDE=inf2 NEURON_RT_VISIBLE_CORES=0 NEURON_SCRATCHPAD_PAGE_SIZE=2048"
echo "  python -m boltz_neuron predict input.yaml --out_dir out --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 --num_workers 0"
echo ""
echo "A shape that is not in the cache compiles on first use; an inf2.xlarge does not have enough"
echo "RAM for that. Build caches for every shape you serve on an inf2.8xlarge (build_cache.sh)."
