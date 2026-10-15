#!/bin/bash
# One SAGE run on Video (k = 3, theta = 0.3, as in Table 8) to check the installation end to end.
#   bash ARI/scripts/smoke_test.sh
ARI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(dirname "$ARI_DIR")"
source "$ROOT_DIR/paper_figures/experiments_env.sh"   # cds to the repo root, defines run_pipeline

OUT=result/ari_smoke/
rm -rf "$OUT"
start=$SECONDS
DO_PARALLEL=false run_pipeline ./src/models/greedy_basic.py Video 3 0.3 "$OUT"
echo "elapsed: $((SECONDS - start)) s"

div=$(ls -d "$OUT"diversity/data_num* 2>/dev/null | head -1)
if [ -z "$div" ] || [ ! -s "$div/diversity_result.csv" ] || [ ! -s "$div/selected_queries.json" ]; then
  echo "SMOKE TEST FAILED: no diversity results under $OUT; see the newest log/paper_result_ari_smoke_*.txt" >&2
  exit 1
fi
echo "selected pivot tables ($div/selected_queries.json):"
python - "$div/selected_queries.json" <<'EOF'
import json, sys
for i, t in enumerate(json.load(open(sys.argv[1])), 1):
    groups = ", ".join([t["index"]] + ([t["column"]] if t["column"] else []))
    print(f"  {i}. {t['aggfunc'].upper()}({t['value']}) GROUP BY {groups}")
EOF
echo "utility and distance per table ($div/diversity_result.csv):"
sed 's/^/  /' "$div/diversity_result.csv"
echo "SMOKE TEST PASSED (expect 3 pivot tables over Video with minimum pairwise distance >= 0.3)"
