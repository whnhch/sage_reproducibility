#!/bin/bash
# Optional: experiments that need an NVIDIA GPU, then redraw the tables, figures and report.
#   bash ARI/run_optional.sh MODE... [--items LIST] [--datasets LIST] [--fig10-parts LIST] [--iters LIST]
#   --bf           BruteForce in Table 9 and Fig. 10 left (calls Llama)
#   --llama        variants without the LLM-proxy (call Llama)
#   --gpu          rerun with live T5 / TAPEX encoding, so runtimes include it
#   --proxy        regenerate the LLM-proxies into result/proxy/ (only --datasets applies)
#   e.g. bash ARI/run_optional.sh --llama --items table9
#        bash ARI/run_optional.sh --proxy --datasets Video
# Llama needs hf_token.txt. Finished runs are skipped (except with --gpu).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

BF=0 LLAMA=0 LIVE=0 PROXY=0 ITEMS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bf)          BF=1 ;;
    --llama)       LLAMA=1 ;;
    --gpu)         LIVE=1 ;;
    --proxy)       PROXY=1 ;;
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
    -h|--help)     usage; exit 0 ;;
    *)             echo "unknown option '$1' (see --help)" >&2; exit 1 ;;
  esac
  shift
done
[ $((BF + LLAMA + LIVE + PROXY)) -gt 0 ] || { usage; exit 1; }

activate
gpu_visible || { echo "No GPU visible to torch: these experiments need an NVIDIA GPU." >&2; exit 1; }
if [ "$BF" = 1 ] || [ "$LLAMA" = 1 ] || [ "$PROXY" = 1 ]; then
  token="${HF_TOKEN_PATH:-$ROOT_DIR/hf_token.txt}"
  [ -s "$token" ] || { echo "$token is missing or empty: Llama-3 needs a Hugging Face token with access to it." >&2; exit 1; }
  stage "Llama-3 -> $TRANSFORMER_PATH"
  python "$ARI_DIR/scripts/fetch_models.py" --llama || exit 1
fi

if [ "$PROXY" = 1 ]; then
  stage "LLM-proxies: Llama labels, then correlation / ratio / outlier trees -> result/proxy/"
  bash src/prompt_learning/regen_proxies.sh || echo "some proxies failed; see log/proxy_*.txt" >&2
  echo "Rerun the experiments with the new trees: FORCE=1 PROXY_DIR=result/proxy/model bash ARI/run_experiments.sh"
  [ $((BF + LLAMA + LIVE)) -gt 0 ] || exit 0
fi

export WITH_BF=$BF WITH_NO_PROXY=$LLAMA EMBEDDINGS=bank
if [ "$LIVE" = 1 ]; then
  export EMBEDDINGS=live FORCE=1   # replace the bank-mode runs with live-encoded ones
  ITEMS="${ITEMS:-table8 table9 fig10 fig11}"
else
  ITEMS="${ITEMS:-table9 fig10}"
  export FIG10_PARTS="${FIG10_PARTS:-left}"   # these variants are only in the left part
fi
stage "experiments: $ITEMS (BruteForce $BF, no-proxy variants $LLAMA, live encoding $LIVE)"
# shellcheck disable=SC2086
SKIP_PDFS=1 bash paper_figures/run_all.sh $ITEMS || echo "some runs failed; see log/paper_run_*.txt" >&2
draw_figures
