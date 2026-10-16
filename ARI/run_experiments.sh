#!/bin/bash
# Step 2: run the default experiments (no GPU), then draw the tables, figures and report.
#   bash ARI/run_experiments.sh [--items LIST] [--datasets LIST] [--fig10-parts LIST] [--iters LIST]
#   --figures-only  only draw (needs TeX Live)
#   --paper-text    pruning threshold 0.5 everywhere (the paper text's), separate outputs;
#                   first, on a GPU: PARAMS=paper_text python -m src.diversity.embedding_bank
#   e.g. --items fig11 --datasets covertype_expanded
# FORCE=1 reruns finished runs. GPU experiments: ARI/run_optional.sh.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

ITEMS="table8 table9 fig10 fig11" RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --items|--datasets|--fig10-parts|--iters)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 1; }
      value="${2//,/ }"
      case "$1" in
        --items)       ITEMS="$value" ;;
        --datasets)    export DATASETS="$value" ;;
        --fig10-parts) export FIG10_PARTS="$value" ;;
        --iters)       export ITERS="$value" ;;
      esac
      shift ;;
    --figures-only) RUN=0 ;;
    --paper-text)   export PARAMS=paper_text ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "unknown option '$1' (see --help)" >&2; exit 1 ;;
  esac
  shift
done

activate
if [ "$RUN" = 1 ]; then
  export WITH_BF=0 WITH_NO_PROXY=0 EMBEDDINGS=bank   # the default experiments; the rest is in run_optional.sh
  stage "experiments: $ITEMS"
  # shellcheck disable=SC2086
  SKIP_PDFS=1 bash paper_figures/run_all.sh $ITEMS || echo "some runs failed; see log/paper_run_*.txt" >&2
fi
draw_figures
