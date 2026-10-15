#!/bin/bash
# Raw results for Table 9: BruteForce and the PR/PX ablation on the first 5 attributes of
# Marketing, theta = 0.1, k = 2 and 5. Writes result/paper/table9/{bruteforce,sage}/...
#
# By default only the variants with the LLM-proxy run (No PR, SAGE): a few minutes. Two switches add the rest:
#   WITH_BF=1        BruteForce: enumerates C(130, k) sets, about 1.6 h for k = 2 and 5.4 h for k = 5 (paper runtimes)
#   WITH_NO_PROXY=1  No PR/PX and No PX: call Llama on the GPU for every candidate, about 1.4 h and 8 min per k
# Cells of skipped variants are filled from the paper by make_table9.py and listed in table9_provenance.csv.
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
