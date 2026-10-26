#!/bin/bash
# Throughput benchmark: one worker per logical NeuronCore (DP), each predicting copies of the same input in ONE
# process (model loaded once, as in serving).
#   bash scripts/bench_dp.sh <input.yaml> <diffusion_samples> [workers=4] [reps=3]
# Phase A: one worker predicts one copy to warm the compile caches (not timed). Phase B: all workers run at once,
# each predicting reps+1 copies. Per-structure time = gap between consecutive output writes (the first copy, which
# includes model load, is skipped). Prints per-worker median seconds and
# structures/hour = workers * ds * 3600 / (slowest worker median).
# Env: BOLTZ_NEURON_STEERING (off | on | nim) and the other BOLTZ_NEURON_* knobs; RECYCLING (3), STEPS (50).
set -u
IN=${1:?input yaml}; DS=${2:?diffusion samples}; W=${3:-4}; REPS=${4:-3}
OUT=${OUT:-bench_dp_out}; mkdir -p "$OUT"
NAME=$(basename "$IN" .yaml); TAG="${NAME}_ds${DS}"
run(){  # core ncopies tag
  local core=$1 n=$2 tag=$3 d="$OUT/${3}_c${1}"
  rm -rf "$d" && mkdir -p "$d/in"
  for i in $(seq 0 $((n-1))); do cp "$IN" "$d/in/${NAME}_$i.yaml"; done
  NEURON_RT_VISIBLE_CORES=$core python -m boltz_neuron predict "$d/in" --out_dir "$d/out" \
    --recycling_steps "${RECYCLING:-3}" --sampling_steps "${STEPS:-50}" --diffusion_samples "$DS" \
    --num_workers 0 --override > "$d.log" 2>&1 || echo "core $core FAILED (see $d.log)"; }
echo "[A] warm caches on core 0"; run 0 1 "${TAG}_warm"
echo "[B] $W workers x $((REPS+1)) predictions"
for c in $(seq 0 $((W-1))); do run "$c" $((REPS+1)) "$TAG" & done; wait
python3 - "$OUT" "$TAG" "$W" "$DS" "$NAME" <<'PY'
import sys, glob, os, statistics as st, json
out, tag, w, ds, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
med = {}
for d in sorted(glob.glob(f"{out}/{tag}_c[0-9]*")):
    if not os.path.isdir(d): continue
    cifs = glob.glob(f"{d}/out/**/{name}_*_model_0.cif", recursive=True)
    t = sorted(os.path.getmtime(f) for f in cifs)
    gaps = [b - a for a, b in zip(t, t[1:])]
    if gaps: med[d.rsplit("_c", 1)[1]] = round(st.median(gaps), 2)
if len(med) < w:
    print("RESULT " + json.dumps({"tag": tag, "error": f"{len(med)}/{w} workers produced timings", "per_worker_median_s": med})); sys.exit(1)
worst = max(med.values())
print("RESULT " + json.dumps({"tag": tag, "workers": w, "ds": ds, "per_worker_median_s": med, "slowest_s": worst,
                              "structures_per_hour": round(w * ds * 3600 / worst, 1)}))
PY
