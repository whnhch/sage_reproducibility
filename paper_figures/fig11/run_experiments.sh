#!/bin/bash
# Raw results for Fig. 11: SAGE (greedy_basic.py) and SAGE+ (greedy.py) runtime on Marketing
# and CoverType while varying #tuples (data_num) and #attributes (max_columns); k = 5,
# theta = 0.1. Writes result/paper/fig11/{tuples,attributes}/{sage,sage_plus}/...
source "$(dirname "${BASH_SOURCE[0]}")/../experiments_env.sh"
OUT="$OUT_ROOT/fig11"
K=5
TH=0.1

n_columns() { case "$1" in marketing_data_full) echo 28 ;; covertype_expanded) echo 110 ;; esac; }

for ITER in $ITERS; do
  for DS in ${DATASETS:-marketing_data_full covertype_expanded}; do   # override with DATASETS="..."
    for PCT in 20 40 60 80 100; do
      FRAC=$(python3 -c "print($PCT/100)")
      COLS=$(python3 -c "print(round($PCT/100*$(n_columns "$DS")))")
      # tuples: first FRAC of the rows, all columns
      DATA_NUM=$FRAC run_pipeline ./src/models/greedy_basic.py "$DS" $K $TH "$OUT/tuples/sage/$DS/$(iter_dir "$ITER")"
      DATA_NUM=$FRAC run_pipeline ./src/models/greedy.py       "$DS" $K $TH "$OUT/tuples/sage_plus/$DS/$(iter_dir "$ITER")"
      # attributes: all rows, first COLS columns
      DATA_NUM=1.0 MAX_COLS=$COLS run_pipeline ./src/models/greedy_basic.py "$DS" $K $TH "$OUT/attributes/sage/$DS/$(iter_dir "$ITER")"
      DATA_NUM=1.0 MAX_COLS=$COLS run_pipeline ./src/models/greedy.py       "$DS" $K $TH "$OUT/attributes/sage_plus/$DS/$(iter_dir "$ITER")"
    done
  done
done

echo "done -> $OUT ; now run: python paper_figures/fig11/make_fig11.py --source results"
