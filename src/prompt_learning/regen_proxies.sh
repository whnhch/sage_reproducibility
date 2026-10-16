#!/bin/bash
# Regenerate the LLM-proxy trees: Llama-3 labels (GPU), then one tree per dataset and kind, into $OUT.
#   DATASETS="Video house" N=10000 OUT=result/proxy bash src/prompt_learning/regen_proxies.sh
# Use them: FORCE=1 PROXY_DIR=result/proxy/model bash ARI/run_experiments.sh
source "$(dirname "${BASH_SOURCE[0]}")/../../paper_figures/experiments_env.sh"

DATASETS="${DATASETS:-marketing_data_full Video house covertype_expanded}"
N="${N:-10000}"
OUT="${OUT:-result/proxy}"
KINDS="correlation ratio outlier"
FAILED=0

for DS in $DATASETS; do
  dataset_config "$DS" || exit 1
  echo "[$(date +%H:%M:%S)] $DS: labeling $N configs each for $KINDS with Llama-3 -> $OUT/data/"
  # shellcheck disable=SC2086
  python -B -m src.prompt_learning.generate_dataset --dataset "$DS" --filepath ./data/ --kinds $KINDS \
    --max_columns "$MAX_COLUMNS" --n "$N" --seed "$SEED" --out_dir "$OUT/data" \
    --transformer_path "$TRANSFORMER_PATH" --hf_token_path "$HF_TOKEN_PATH" \
    > "log/proxy_data_${DS}.txt" 2>&1 || { echo "  FAILED, see log/proxy_data_${DS}.txt" >&2; FAILED=1; continue; }
  for KIND in $KINDS; do
    model="$OUT/model/${DS}_${N}_${KIND}_decision_tree.pkl"
    if python -B -m src.prompt_learning.train --data "$OUT/data/${DS}_${KIND}.jsonl" --kind "$KIND" \
         --n "$N" --seed "$SEED" --out "$model" > "log/proxy_train_${DS}_${KIND}.txt" 2>&1; then
      echo "  $KIND: $(grep 'test accuracy' "log/proxy_train_${DS}_${KIND}.txt") -> $model"
    else
      echo "  $KIND: FAILED, see log/proxy_train_${DS}_${KIND}.txt" >&2; FAILED=1
    fi
  done
done

# test accuracy summary
python3 - "$OUT/model" "$N" $DATASETS <<'PY'
import csv, os, sys
model_dir, n, datasets = sys.argv[1], sys.argv[2], sys.argv[3:]
kinds = ["correlation", "ratio", "outlier"]
rows = []
for ds in datasets:
    row = {"dataset": ds}
    for kind in kinds:
        path = os.path.join(model_dir, f"{ds}_{n}_{kind}_decision_tree_report.csv")
        if os.path.exists(path):
            acc = {r[""]: r for r in csv.DictReader(open(path))}.get("accuracy")
            row[kind] = f"{float(acc['precision']):.3f}" if acc else ""
    rows.append(row)
if any(len(r) > 1 for r in rows):
    with open(os.path.join(model_dir, "accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset"] + kinds)
        w.writeheader()
        w.writerows(rows)
    print(f"\ntest accuracy (paper, Section 6.2: correlation 90%, ratio 89%, surprise 64% on average) -> {model_dir}/accuracy.csv")
    for r in rows:
        print(f"  {r['dataset']:20s} " + "  ".join(f"{k} {r.get(k) or '-':>5s}" for k in kinds))
PY
exit $FAILED
