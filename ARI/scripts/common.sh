#!/bin/bash
# Shared helpers for ARI/install.sh, ARI/run_experiments.sh and ARI/run_optional.sh (sourced, not run).

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/$(basename "${BASH_SOURCE[1]}")"   # the sourcing script
ARI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(dirname "$ARI_DIR")"
cd "$ROOT_DIR" || exit 1
mkdir -p log
export TRANSFORMER_PATH="${TRANSFORMER_PATH:-$HOME/.cache/huggingface}"

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }

usage() {  # the comment block at the top of the calling script
  sed -n '2,/^[^#]/p' "$SCRIPT" | sed '$d; s/^# \{0,1\}//'
}

env_ok() {  # the active Python matches ARI/env/requirements.txt (torch: 2.6 or newer, which newer GPUs need)
  python - "$ARI_DIR/env/requirements.txt" <<'PY'
import sys
from importlib import metadata
bad = [] if sys.version_info[:2] == (3, 10) else [f"Python {sys.version.split()[0]} (need 3.10)"]
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if "==" not in line:
        continue
    name, want = (x.strip() for x in line.split("=="))
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        bad.append(f"{name} missing (need {want})")
        continue
    if name == "torch":
        ok = [int(x) for x in got.split("+")[0].split(".")[:2]] >= [int(x) for x in want.split(".")[:2]]
    else:
        ok = got == want
    if not ok:
        bad.append(f"{name} {got} (need {want})")
for b in bad:
    print("  " + b, file=sys.stderr)
sys.exit(1 if bad else 0)
PY
}

SAGE_ENV="${SAGE_ENV:-sage}"   # conda environment: a name or a path (conda env create -p)

conda_hook() {
  command -v conda >/dev/null 2>&1 || {
    echo "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html), then: bash ARI/install.sh" >&2
    exit 1
  }
  eval "$(conda shell.bash hook)"
}

activate() {  # the pinned conda environment SAGE_ENV, whatever is active
  conda_hook
  conda activate "$SAGE_ENV" 2>/dev/null || { echo "conda environment '$SAGE_ENV' not found: run 'bash ARI/install.sh'" >&2; exit 1; }
  env_ok || { echo "conda environment '$SAGE_ENV' does not match ARI/env/requirements.txt (above): run 'bash ARI/install.sh'" >&2; exit 1; }
}

gpu_visible() {
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

have_latex() {  # pdflatex with the packages the tables and figures use
  command -v pdflatex >/dev/null 2>&1 || return 1
  local f
  for f in standalone.cls libertine.sty newtxmath.sty pgfplots.sty booktabs.sty pdfpages.sty; do
    command -v kpsewhich >/dev/null 2>&1 && [ -z "$(kpsewhich "$f")" ] && return 1
  done
  return 0
}

draw_figures() {  # Tables 8-9, Figs. 10-11 from result/paper/, then ARI/out/reproduced_results.pdf
  local RES=() FIGDIR=paper_figures/out REPORT=ARI/out/reproduced_results.pdf
  if [ "${PARAMS:-code}" = paper_text ]; then   # see PARAMS in paper_figures/experiments_env.sh
    FIGDIR=paper_figures/out_paper_text REPORT=ARI/out/reproduced_results_paper_text.pdf
    RES=(--results result/paper_text --out "$FIGDIR")
  fi
  if ! have_latex; then
    echo "LaTeX (pgfplots, libertine, newtx, standalone, pdfpages) not found: skipping the figures."
    echo "Results are in ${RES[1]:-result}/paper/. Draw them on a machine with a full TeX Live:"
    echo "  bash ARI/run_experiments.sh --figures-only"
    return 0
  fi
  stage "figures and report"
  local item
  for item in table8 table9 fig10 fig11; do
    python -B "paper_figures/$item/make_$item.py" --source results ${RES[@]+"${RES[@]}"} || echo "$item failed" >&2
  done
  python -B "$ARI_DIR/scripts/make_report.py" --figures-dir "$FIGDIR" --output "$REPORT" && echo "summary: $REPORT"
}
