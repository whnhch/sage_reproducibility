#!/bin/bash
# Table 8 runs -> result/paper/table8/. SAGE = greedy_basic.py, SAGE+ = greedy.py, Top-k = SAGE with theta 0;
# the baseline lists in baselines/ are scored with software.py (REGEN_BASELINES=1 regenerates LLM and DAISY).
source "$(dirname "${BASH_SOURCE[0]}")/../experiments_env.sh"
OUT="$OUT_ROOT/table8"

DATASETS="${DATASETS:-marketing_data_full Video house}"   # override, e.g. DATASETS="Video"
theta_for() {  # theta for (dataset, k)
  if [ "$2" = "10" ]; then echo 0.1; elif [ "$1" = "house" ]; then echo 0.2; else echo 0.3; fi
}

for ITER in $ITERS; do
  for DS in $DATASETS; do
    for K in 3 5 10; do
      TH=$(theta_for "$DS" "$K")
      DO_PARALLEL=false run_pipeline ./src/models/greedy_basic.py "$DS" "$K" "$TH" "$OUT/sage/$DS/$(iter_dir "$ITER")"
      DO_PARALLEL=true  run_pipeline ./src/models/greedy.py       "$DS" "$K" "$TH" "$OUT/sage_plus/$DS/$(iter_dir "$ITER")"
      DO_PARALLEL=false run_pipeline ./src/models/greedy_basic.py "$DS" "$K" 0.0   "$OUT/topk/$DS/$(iter_dir "$ITER")"
    done
  done
done

# ---- baselines ---------------------------------------------------------------
if [ "${REGEN_BASELINES:-0}" = "1" ]; then
  for DS in $DATASETS; do
    dataset_config "$DS"
    for K in 3 5 10; do
      python -B ./src/models/llm.py --filename "$DS" --filepath ./data/ --k "$K" \
        --transformer_path "$TRANSFORMER_PATH" --hf_token_path "$HF_TOKEN_PATH" --seed "$SEED" \
        --data_num 10 --max_columns "$MAX_COLUMNS" --output_dir "$OUT/llm_raw/$DS/$K/" \
        > "log/paper_llm_${DS}_${K}.txt" 2>&1
      cp "$(ls "$OUT"/llm_raw/"$DS"/"$K"/diversity/*/best_set.csv | head -1)" "baselines/llm_${K}_${DS}.csv"
    done
    python -B ./src/models/daisy.py --filename "$DS" --filepath ./data/ --model_path ./daisy.pth \
      --transformer_path "$TRANSFORMER_PATH" --hf_token_path "$HF_TOKEN_PATH" --seed "$SEED" \
      --gamma "$GAMMA" --k 20 --data_num "$DATA_FRAC" --max_columns "$MAX_COLUMNS" \
      --output_dir "$OUT/daisy_raw/$DS/" > "log/paper_daisy_${DS}.txt" 2>&1
    cp "$(ls "$OUT"/daisy_raw/"$DS"/diversity/*/top_20_combinations.csv | head -1)" "baselines/daisy_${DS}.csv"
  done
fi

# baselines/ file names are not uniform (e.g. excel_video.csv, daisy_Video.csv); pick whichever exists
comb_name() {  # <tool> <dataset> [k]
  local tool=$1 ds=$2 k=$3 lower short
  lower=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
  short=${lower%_data_full}
  for c in "${tool}${k:+_$k}_${ds}" "${tool}${k:+_$k}_${lower}" "${tool}${k:+_$k}_${short}"; do
    [ -f "baselines/$c.csv" ] && { echo "$c"; return; }
  done
  echo "${tool}${k:+_$k}_${ds}"
}

for DS in $DATASETS; do
  for K in 3 5 10; do
    for TOOL in daisy powerbi gsheets excel; do
      frac=""; [ "$TOOL" = powerbi ] && [ "$DS" = house ] && frac=0.1
      score_recommendations "$(comb_name "$TOOL" "$DS")" "$DS" "$K" "$OUT/$TOOL/" $frac
    done
    score_recommendations "$(comb_name llm "$DS" "$K")" "$DS" "$K" "$OUT/llm_k$K/"
  done
done

echo "done -> $OUT ; now run: python paper_figures/table8/make_table8.py --source results"
