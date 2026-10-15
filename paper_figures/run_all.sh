#!/bin/bash
# Run the experiments for Table 8, Table 9, Fig. 10, Fig. 11 and then build all four PDFs
# from the new results.
#
#   bash paper_figures/run_all.sh                  # everything
#   bash paper_figures/run_all.sh fig11 table8     # only these artifacts (in this order)
#
# Settings from experiments_env.sh can be overridden here too, e.g.
#   TRANSFORMER_PATH=~/hf_cache ITERS="0" bash paper_figures/run_all.sh
# Each artifact's console output is saved to log/paper_run_<item>.txt.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$HERE")"
cd "$ROOT_DIR" || exit 1
mkdir -p log
RUN_TAG=""; [ "${PARAMS:-code}" = paper_text ] && RUN_TAG=paper_text_   # see PARAMS in experiments_env.sh

ITEMS=("$@")
[ ${#ITEMS[@]} -eq 0 ] && ITEMS=(table8 table9 fig10 fig11)

failed=()
for item in "${ITEMS[@]}"; do
  if [ ! -f "$HERE/$item/run_experiments.sh" ]; then
    echo "unknown artifact '$item' (choose from table8 table9 fig10 fig11)" >&2
    exit 1
  fi
  echo "===== [$(date '+%F %T')] experiments for $item ====="
  start=$SECONDS
  if bash "$HERE/$item/run_experiments.sh" 2>&1 | tee "log/paper_run_${RUN_TAG}${item}.txt"; then
    echo "===== $item finished in $(( (SECONDS - start) / 60 )) min ====="
  else
    echo "===== $item exited with an error, see log/paper_run_${RUN_TAG}${item}.txt =====" >&2
    failed+=("$item")
  fi
  # individual runs that fail are reported as "FAILED, see log/..." without stopping the loop
  grep -h "FAILED" "log/paper_run_${RUN_TAG}${item}.txt" >&2 && failed+=("$item (some runs)")
done

latex_ok=1  # drawing needs pdflatex with these packages; skip it (not an error) where they are missing
[ "${SKIP_PDFS:-0}" = 1 ] && latex_ok=skip   # ARI/run_experiments.sh draws everything itself afterwards
command -v pdflatex >/dev/null 2>&1 || latex_ok=0
for f in standalone.cls libertine.sty newtxmath.sty pgfplots.sty; do
  command -v kpsewhich >/dev/null 2>&1 && [ -z "$(kpsewhich "$f")" ] && latex_ok=0
done
echo
if [ "$latex_ok" = skip ]; then
  :
elif [ "$latex_ok" = 1 ]; then
  echo "===== [$(date '+%F %T')] building PDFs from results ====="
  for item in "${ITEMS[@]}"; do
    python3 "$HERE/$item/make_$item.py" --source results || failed+=("$item (pdf)")
  done
  echo
  echo "PDFs: $HERE/out/   (per-cell sources: $HERE/out/*_provenance.csv)"
else
  echo "Skipping the PDFs: this machine's LaTeX lacks standalone/libertine/newtx/pgfplots."
  echo "Results are in result/paper/. Draw them on a machine with a full TeX Live:"
  echo "  bash ARI/run_experiments.sh --figures-only"
fi
if [ ${#failed[@]} -gt 0 ]; then
  echo "Problems: ${failed[*]}" >&2
  exit 1
fi
