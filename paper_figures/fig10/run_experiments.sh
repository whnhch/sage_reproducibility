#!/bin/bash
# Raw results for Fig. 10.
#   left : BruteForce + PR/PX ablation, k = 4, theta = 0.1, first 5 attributes of
#          Marketing, Video and House -> result/paper/fig10/{bruteforce,ablation}/...
#   right: SAGE on Video for k in {5,10,15,20} and theta in {0.0,...,1.0} -> result/paper/fig10/theta/...
#
# By default the left plot runs only the variants with the LLM-proxy (-PR, SAGE). Two switches add the rest:
#   WITH_BF=1        BruteForce bars: about 2.3 h on Marketing, 8 min on Video, 36 min on House (paper runtimes)
#   WITH_NO_PROXY=1  -PR&PX and -PX bars: Llama on the GPU for every candidate, about 1.5 h on Marketing,
#                    2 min on Video, 17 min on House
# Bars of skipped variants are taken from the paper by make_fig10.py and listed in fig10_provenance.csv.
source "$(dirname "${BASH_SOURCE[0]}")/../experiments_env.sh"
OUT="$OUT_ROOT/fig10"
WITH_BF="${WITH_BF:-0}"
WITH_NO_PROXY="${WITH_NO_PROXY:-0}"
FIG10_PARTS="${FIG10_PARTS:-left right}"   # e.g. FIG10_PARTS=left to rerun only the ablation bars
[ "$WITH_BF" = 1 ] || echo "skipping BruteForce (set WITH_BF=1 to run it)"
[ "$WITH_NO_PROXY" = 1 ] || echo "skipping -PR&PX and -PX (set WITH_NO_PROXY=1 to run them)"

# ---- left ----
# Pruning threshold of the ablation: 0.5 for Marketing, 0.3 for Video and House.
ablation_int_threshold() {
  if [ "$1" = "marketing_data_full" ]; then echo "${ABLATION_INT_THRESHOLD:-0.5}"; else echo "${FIG10_SMALL_INT_THRESHOLD:-0.3}"; fi
}
for DS in ${DATASETS:-marketing_data_full Video house}; do   # override with DATASETS="..."
  [[ " $FIG10_PARTS " == *" left "* ]] || break
  if [ "$WITH_BF" = 1 ]; then
    MAX_COLS=5 DO_PRUNE=false DO_CACHE=false INT_TH=0.0 \
      run_pipeline ./src/models/bruteforce.py "$DS" 4 0.1 "$OUT/bruteforce/$DS/"
  fi
  for ITER in $ITERS; do
    for FLAGS in "false false" "false true" "true false" "true true"; do   # -PR&PX, -PR, -PX, SAGE
      set -- $FLAGS
      [ "$2" = false ] && [ "$WITH_NO_PROXY" != 1 ] && continue   # without the proxy: Llama for every candidate
      MAX_COLS=5 INT_TH="$(ablation_int_threshold "$DS")" DO_PRUNE=$1 DO_CACHE=$2 \
        run_pipeline ./src/models/greedy_basic.py "$DS" 4 0.1 "$OUT/ablation/$DS/$(iter_dir "$ITER")"
    done
  done
done

# ---- right ----
for K in 5 10 15 20; do
  [[ " $FIG10_PARTS " == *" right "* ]] || break
  for TH in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    run_pipeline ./src/models/greedy_basic.py Video "$K" "$TH" "$OUT/theta/Video/"
  done
done

echo "done -> $OUT ; now run: python paper_figures/fig10/make_fig10.py --source results"
