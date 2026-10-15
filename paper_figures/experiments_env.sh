#!/bin/bash
# Shared settings for paper_figures/*/run_experiments.sh.
# Override any of them from the environment, e.g. TRANSFORMER_PATH=~/hf_cache bash table8/run_experiments.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

TRANSFORMER_PATH="${TRANSFORMER_PATH:-$HOME/.cache/huggingface}"   # model cache (Hugging Face default)
HF_TOKEN_PATH="${HF_TOKEN_PATH:-./hf_token.txt}"
export HF_HOME="$TRANSFORMER_PATH"
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"

SEED=42
TAU_COR=0.8
TAU_RAT=5.0
ITERS="${ITERS:-0}"              # one pass: runs with the LLM-proxy are seeded and repeat exactly; ITERS="0 1 2" averages three
iter_dir() { [ "$1" = 0 ] || echo "iter$1/"; }   # sub-folder of repetition $1 (none for the first pass)
method_of() {  # output name of a run script: SAGE = greedy_basic.py, SAGE+ = greedy.py
  case "$(basename "$1" .py)" in greedy_basic) echo sage ;; greedy) echo sage_plus ;; *) basename "$1" .py ;; esac
}
# PARAMS=paper_text: tau_rho 0.5, tau_pi 2.0, pruning threshold 0.5 everywhere; separate results and logs.
PARAMS="${PARAMS:-code}"
LOG_TAG=""
if [ "$PARAMS" = paper_text ]; then
  TAU_COR=0.5 TAU_RAT=2.0 LOG_TAG=paper_text_
  export FIG10_SMALL_INT_THRESHOLD=0.5 COVERTYPE_INT_THRESHOLD=0.5
  OUT_ROOT="${OUT_ROOT:-result/paper_text/paper}"
elif [ "$PARAMS" != code ]; then
  echo "PARAMS must be code or paper_text" >&2; exit 1
fi
OUT_ROOT="${OUT_ROOT:-result/paper}"
PROXY_DIR="${PROXY_DIR:-./src/prompt_learning/model}"   # LLM-proxy decision trees
# Every run goes through src/models/run_cached.py: Llama-3 is loaded only if the run calls it (no LLM-proxy,
# BruteForce), and EMBEDDINGS=bank (default) reads the T5 / TAPEX vectors from precompute/embeddings/
# instead of encoding them. EMBEDDINGS=live encodes them (GPU), so runtimes include encoding.
EMBEDDINGS="${EMBEDDINGS:-bank}"
mkdir -p log "$OUT_ROOT"

# Per-dataset settings (gamma, LLM sample size, row fraction, column cap, interpretability threshold)
dataset_config() {
  case "$1" in
    marketing_data_full) GAMMA=180000000.0; DF_SAMPLE_NUM=5;  DATA_FRAC=1.0; MAX_COLUMNS=90;  INT_THRESHOLD=0.5 ;;
    Video)               GAMMA=500000.0;    DF_SAMPLE_NUM=90; DATA_FRAC=0.1; MAX_COLUMNS=90;  INT_THRESHOLD=0.5 ;;
    house)               GAMMA=500000000.0; DF_SAMPLE_NUM=90; DATA_FRAC=1.0; MAX_COLUMNS=90;  INT_THRESHOLD=0.5 ;;
    covertype_expanded)  GAMMA=180000000.0; DF_SAMPLE_NUM=5;  DATA_FRAC=1.0; MAX_COLUMNS=200; INT_THRESHOLD=${COVERTYPE_INT_THRESHOLD:-0.6} ;;
    *) echo "unknown dataset $1" >&2; return 1 ;;
  esac
}

# run_pipeline <script.py> <dataset> <k> <theta> <output_dir>
# Optional env overrides per call: DO_PRUNE DO_CACHE DO_INT_CACHE DO_PARALLEL DATA_NUM MAX_COLS INT_TH
#   DO_CACHE=true  -> LLM-proxy decision trees instead of Llama calls (the "PX" optimization)
#   DO_PRUNE=true  -> offline pruning (the "PR" optimization)
# run_done <output_dir> <dataset> <k> <theta> <data_num> <max_columns> <int_threshold> <do_prune> <do_cache> <do_parallel>
# True if that run already finished: its diversity folder, named like src/helper.py::param_dict_to_folder,
# holds diversity_result.csv.
run_done() {
  python3 - "$@" <<'PY'
import os, sys
out, ds, k, theta, dn, mc, it, prune, cache, par = sys.argv[1:]
b = lambda v: "True" if v.lower() in ("yes", "true", "t", "y", "1") else "False"
params = {"filename": ds, "k": int(k), "threshold": float(theta), "int_threshold": float(it), "do_prune": b(prune),
          "data_num": float(dn), "do_parallel": b(par), "do_cache": b(cache), "max_columns": int(mc)}
name = "_".join(f"{key}{str(v).replace('.', '-')}" for key, v in sorted(params.items()))
sys.exit(0 if os.path.exists(os.path.join(out, "diversity", name, "diversity_result.csv")) else 1)
PY
}

run_pipeline() {
  local script=$1 ds=$2 k=$3 theta=$4 out=$5 method
  method=$(method_of "$script")
  dataset_config "$ds" || return 1
  if [ "${FORCE:-0}" != 1 ] && run_done "$out" "$ds" "$k" "$theta" "${DATA_NUM:-$DATA_FRAC}" "${MAX_COLS:-$MAX_COLUMNS}" \
       "${INT_TH:-$INT_THRESHOLD}" "${DO_PRUNE:-true}" "${DO_CACHE:-true}" "${DO_PARALLEL:-false}"; then
    echo "[skip] $method $ds k=$k theta=$theta prune=${DO_PRUNE:-true} proxy=${DO_CACHE:-true}: already done (FORCE=1 reruns)"
    return 0
  fi
  local extra=()
  # bruteforce.py has no --do_int_cache flag
  [ "$(basename "$script")" != "bruteforce.py" ] && extra=(--do_int_cache "${DO_INT_CACHE:-true}")
  local log_file="log/paper_${LOG_TAG}$(echo "${out#"$OUT_ROOT"/}" | tr '/' '_')${method}_k${k}_th${theta}_pr${DO_PRUNE:-true}_px${DO_CACHE:-true}_dn${DATA_NUM:-all}_mc${MAX_COLS:-all}.txt"
  echo "[$(date +%H:%M:%S)] $method $ds k=$k theta=$theta prune=${DO_PRUNE:-true} proxy=${DO_CACHE:-true} -> $out"
  python -B -m src.models.run_cached "$(basename "$script" .py)" --embeddings "$EMBEDDINGS" \
    --filename "$ds" --filepath "./data/" \
    --transformer_path "$TRANSFORMER_PATH" --hf_token_path "$HF_TOKEN_PATH" \
    --seed "$SEED" --gamma "$GAMMA" --df_sample_num "$DF_SAMPLE_NUM" \
    --int_threshold "${INT_TH:-$INT_THRESHOLD}" --threshold "$theta" --k "$k" \
    --clf_trd_path "$PROXY_DIR/${ds}_10000_decision_tree.pkl" \
    --clf_out_path "$PROXY_DIR/${ds}_10000_outlier_decision_tree.pkl" \
    --tau_cor "$TAU_COR" --tau_rat "$TAU_RAT" \
    --do_prune "${DO_PRUNE:-true}" --do_cache "${DO_CACHE:-true}" \
    "${extra[@]}" \
    --do_parallel "${DO_PARALLEL:-false}" --do_timeout false \
    --data_num "${DATA_NUM:-$DATA_FRAC}" --max_columns "${MAX_COLS:-$MAX_COLUMNS}" \
    --attribute_aggfunc_ranks "./precompute/${ds}_attribute_aggfunc_ranks.json" \
    --interesting_attributes_path "./precompute/${ds}_interesting_attrs.json" \
    --output_dir "$out" > "$log_file" 2>&1 || echo "  FAILED, see $log_file" >&2
}

# score_recommendations <baselines/csv name without .csv> <dataset> <k> <output_dir> [row fraction]
# Scores a list of recommended pivot tables (Excel/PowerBI/GSheets/LLM/DAISY) with src/models/software.py.
score_recommendations() {
  local comb=$1 ds=$2 k=$3 out=$4
  dataset_config "$ds" || return 1
  local frac="${5:-$DATA_FRAC}" dn
  if [ ! -f "baselines/${comb}.csv" ]; then echo "  skip: baselines/${comb}.csv not found"; return 0; fi
  dn=$(python3 -c "print(str(float('$frac')).replace('.', '-'))")
  local done_file=("$out"diversity/data_num"${dn}"_*_filename"${ds}"_int_thresholdNone_k"${k}"_max_columns*_thresholdNone/combined_sqrt_cosine_distance.csv)
  if [ "${FORCE:-0}" != 1 ] && [ -f "${done_file[0]}" ]; then
    echo "[skip] software.py $comb ($ds, k=$k): already done (FORCE=1 reruns)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] software.py $comb ($ds, k=$k) -> $out"
  python -B -m src.models.run_cached software \
    --filename "$ds" --filepath "./data/" --combname "$comb" --combpath "./baselines/" \
    --transformer_path "$TRANSFORMER_PATH" --hf_token_path "$HF_TOKEN_PATH" \
    --seed "$SEED" --k "$k" --gamma "$GAMMA" --df_sample_num "$DF_SAMPLE_NUM" \
    --clf_trd_path "$PROXY_DIR/${ds}_10000_decision_tree.pkl" \
    --clf_out_path "$PROXY_DIR/${ds}_10000_outlier_decision_tree.pkl" \
    --tau_cor "$TAU_COR" --tau_rat "$TAU_RAT" \
    --data_num "$frac" --max_columns "$MAX_COLUMNS" \
    --attribute_aggfunc_ranks "./precompute/${ds}_attribute_aggfunc_ranks.json" \
    --interesting_attributes_path "./precompute/${ds}_interesting_attrs.json" \
    --output_dir "$out" > "log/paper_${LOG_TAG}software_${comb}_k${k}.txt" 2>&1 \
    || echo "  FAILED, see log/paper_${LOG_TAG}software_${comb}_k${k}.txt" >&2
}
