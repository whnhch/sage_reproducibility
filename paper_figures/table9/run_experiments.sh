#!/bin/bash
# Table 9 runs: PR/PX ablation on Marketing's first 5 attributes, theta 0.1, k 2 and 5 -> result/paper/table9/
#   WITH_BF=1        add BruteForce (hours)
#   WITH_NO_PROXY=1  add No PR/PX and No PX (Llama on the GPU)
# Skipped cells are filled from the paper (table9_provenance.csv).
source "$(dirname "${BASH_SOURCE[0]}")/../experiments_env.sh"
OUT="$OUT_ROOT/table9"
DS=marketing_data_full
TH=0.1
export MAX_COLS=5 INT_TH="${ABLATION_INT_THRESHOLD:-0.5}"
WITH_BF="${WITH_BF:-0}"
WITH_NO_PROXY="${WITH_NO_PROXY:-0}"
[ "$WITH_BF" = 1 ] || echo "skipping BruteForce (set WITH_BF=1 to run it: about 1.6 h for k=2, 5.4 h for k=5)"
[ "$WITH_NO_PROXY" = 1 ] || echo "skipping No PR/PX and No PX (set WITH_NO_PROXY=1 to run them: Llama on the GPU, about 1.5 h per k)"

for K in 2 5; do
  if [ "$WITH_BF" = 1 ]; then
    DO_PRUNE=false DO_CACHE=false INT_TH=0.0 run_pipeline ./src/models/bruteforce.py "$DS" "$K" "$TH" "$OUT/bruteforce/$DS/"
  fi
  for ITER in $ITERS; do
    if [ "$WITH_NO_PROXY" = 1 ]; then
      DO_PRUNE=false DO_CACHE=false run_pipeline ./src/models/greedy_basic.py "$DS" "$K" "$TH" "$OUT/sage/$DS/$(iter_dir "$ITER")"  # No PR/PX
    fi
    DO_PRUNE=false DO_CACHE=true  run_pipeline ./src/models/greedy_basic.py "$DS" "$K" "$TH" "$OUT/sage/$DS/$(iter_dir "$ITER")"  # No PR
    if [ "$WITH_NO_PROXY" = 1 ]; then
      DO_PRUNE=true  DO_CACHE=false run_pipeline ./src/models/greedy_basic.py "$DS" "$K" "$TH" "$OUT/sage/$DS/$(iter_dir "$ITER")"  # No PX
    fi
    DO_PRUNE=true  DO_CACHE=true  run_pipeline ./src/models/greedy_basic.py "$DS" "$K" "$TH" "$OUT/sage/$DS/$(iter_dir "$ITER")"  # SAGE
  done
done

echo "done -> $OUT ; now run: python paper_figures/table9/make_table9.py --source results"
