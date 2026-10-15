#!/bin/bash
# Build table8.pdf, table9.pdf, fig10.pdf, fig11.pdf into paper_figures/out/.
# Extra arguments are passed to every script, e.g.:  bash paper_figures/build_all.sh --source results
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for item in table8 table9 fig10 fig11; do
  python3 "$HERE/$item/make_$item.py" "$@"
done
