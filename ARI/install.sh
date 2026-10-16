#!/bin/bash
# Step 1: create the conda environment SAGE_ENV (default: sage; replaced if present), unpack the data, check it.
# Needs conda; no GPU or token.
#   bash ARI/install.sh [--llama] [--no-smoke]
#   --llama     also download Llama-3 (needs hf_token.txt; only for run_optional.sh)
#   --no-smoke  skip the smoke test
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

LLAMA=() SMOKE=1
for arg in "$@"; do
  case "$arg" in
    --llama)    LLAMA=(--llama) ;;
    --no-smoke) SMOKE=0 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "unknown option '$arg' (see --help)" >&2; exit 1 ;;
  esac
done

stage "new conda environment: $SAGE_ENV"
conda_hook
case "$SAGE_ENV" in */*) WHERE=(-p "$SAGE_ENV") ;; *) WHERE=(-n "$SAGE_ENV") ;; esac
for _ in 1 2 3 4 5; do [ -n "${CONDA_PREFIX:-}" ] && conda deactivate || break; done   # an active env cannot be removed
conda remove "${WHERE[@]}" --all -y >/dev/null 2>&1   # always start from a fresh environment
conda env create "${WHERE[@]}" -f "$ARI_DIR/env/environment.yml" || exit 1
activate

stage "data"
python "$ARI_DIR/scripts/unpack_data.py" || exit 1

stage "models -> $TRANSFORMER_PATH"
python "$ARI_DIR/scripts/fetch_models.py" ${LLAMA[@]+"${LLAMA[@]}"} || exit 1

stage "check"
python "$ARI_DIR/scripts/check_install.py" --online || exit 1

if [ "$SMOKE" = 1 ]; then
  stage "smoke test"
  bash "$ARI_DIR/scripts/smoke_test.sh" || exit 1
fi

echo
echo "Installed. Activate the environment in new shells with: conda activate $SAGE_ENV"
echo "Next: bash ARI/run_experiments.sh"
