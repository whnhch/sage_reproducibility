# paper_figures: Table 8, Table 9, Fig. 10, Fig. 11 of `sage.pdf`

Each artifact has its own folder with three files:

| file | what it does |
|---|---|
| `make_<item>.py` | builds `out/<item>.pdf` (LaTeX → `pdflatex`, Libertine font like the paper) |
| `run_experiments.sh` | reruns the experiments with the code in `src/` and writes raw results to `result/paper/<item>/` |
| `paper_values.json` | the numbers printed in `sage.pdf`, used by default and as a fallback |

`common.py` holds the shared result reader and LaTeX helpers. `experiments_env.sh` holds the shared run settings.

Requirements: rendering needs only Python 3 (standard library) and `pdflatex` with `pgfplots`, `libertine` and `newtx`. Running the experiments needs the paper's environment (GPU, Llama-3, TAPEX/T5 in `TRANSFORMER_PATH`, `hf_token.txt`).

Run everything from the repo root (`/Volumes/u1472329/sage`).

```bash
bash paper_figures/run_all.sh                        # run all experiments, then build all four PDFs from them
bash paper_figures/run_all.sh fig11 table8           # same, only for the named artifacts
bash paper_figures/build_all.sh                      # all five PDFs, numbers as printed in the paper
bash paper_figures/build_all.sh --source results     # all five PDFs, recomputed from result/
```

## Two modes

* **`--source paper`** (default) renders `paper_values.json`. This reproduces the published table or figure.
* **`--source results`** recomputes every number from the run folders under `result/` (`--results <dir>` to point elsewhere). It looks first in `result/paper/<item>/`, which `run_experiments.sh` writes, then in older folders such as `result/greedy/final`.
  * Any value with no matching run falls back to the paper value.
  * In Table 8 those cells print in gray. The figures add a note with the fallback count.
  * `out/<item>_provenance.csv` lists, for every cell, whether it came from `result/` or the paper, and which run folders were averaged.

---

## Table 8: SAGE vs. baselines (Marketing / Video / House, k = 3, 5, 10)

**Prompt**
```bash
python3 paper_figures/table8/make_table8.py                     # -> paper_figures/out/table8.pdf
# regenerate from code:
bash    paper_figures/table8/run_experiments.sh                 # REGEN_BASELINES=1 also reruns llm.py / daisy.py
python3 paper_figures/table8/make_table8.py --source results
```

* θ = 0.30 (Marketing, Video) or 0.20 (House) for k = 3, 5, and θ = 0.10 for k = 10. One pass by default (`ITERS="0 1 2"` averages three).
* **SAGE** runs `src/models/greedy_basic.py` and **SAGE+** runs `src/models/greedy.py`, which samples 20% of the pruned candidates.
* **Top-k** runs `greedy_basic.py` with θ = 0, i.e. no diversity constraint but the same optimizations.
* **DAISY / LLMs / PowerBI / GSheets / Excel**: their recommendation lists in `baselines/*.csv` are scored by `src/models/software.py` (first *k* tables).
  * T(s) for these comes from the paper, because those tools were run by hand.
* **Columns**:
  * Util / Ins / Int are sums over the selected tables (utility = (Ins + Int) / 2).
  * m-dist is the minimum pairwise distance.
  * T is pruning + utility + embedding + selection time.
* **Bold / underline** (best / second-best Util and m-dist per block) are computed, not hard-coded.
* **Heatmap column**: drawn from `selected_distance_matrix.csv` (or `combined_sqrt_cosine_distance.csv` for tools) in results mode. Paper mode has no matrices, so the column stays empty.

## Table 9: ablation of pruning (PR) and LLM-proxy (PX) vs. BruteForce

**Prompt**
```bash
python3 paper_figures/table9/make_table9.py                     # -> paper_figures/out/table9.pdf
bash    paper_figures/table9/run_experiments.sh                 # minutes: No PR + SAGE; WITH_BF=1 / WITH_NO_PROXY=1 add the hours-long variants
python3 paper_figures/table9/make_table9.py --source results
```

* Marketing, first 5 attributes (`--max_columns 5`), θ = 0.1, k = 2 and 5.
* **BruteForce** is `bruteforce.py`. The other variants are `greedy_basic.py` with `--do_prune` / `--do_cache` (LLM-proxy) switched on or off.
* **Utility / Ins / Int** are % of BruteForce.
* **Runtime** is the online phase (utility + embedding + selection); pruning is offline (§6.2).
* **#PT** is the number of candidates whose utility was computed. **#PT combinations** is C(#PT, k).
* The ablation's pruning threshold is 0.5 for Marketing (`ABLATION_INT_THRESHOLD=...` to override) and 0.3 for Video and House (`FIG10_SMALL_INT_THRESHOLD=...`).
* Fig. 10 left uses 0.5 for Marketing (its bars equal Table 9's runtimes) and 0.3 for Video and House, where 0.5 keeps only 2 and 0 candidates on 5 attributes (0.3 keeps 4 and 1). Override the latter with `FIG10_SMALL_INT_THRESHOLD`.

## Fig. 10: (left) optimization ablation runtimes, (right) Utility vs. θ on Video

**Prompt**
```bash
python3 paper_figures/fig10/make_fig10.py                       # -> paper_figures/out/fig10.pdf
bash    paper_figures/fig10/run_experiments.sh
python3 paper_figures/fig10/make_fig10.py --source results
```

* **Left**: k = 4, θ = 0.1, first 5 attributes of Marketing, Video and House. Same variants and online runtime as Table 9, log scale.
* **Right**: `greedy_basic.py` on Video, k ∈ {5, 10, 15, 20}, θ ∈ {0.0, 0.1, …, 1.0}.
* The paper prints no labels on the right panel. Its `paper_values.json` values were **read off the plot at 400 dpi** (about ±0.1).

## Fig. 11: scalability in #tuples and #attributes

**Prompt**
```bash
python3 paper_figures/fig11/make_fig11.py                       # -> paper_figures/out/fig11.pdf
bash    paper_figures/fig11/run_experiments.sh
python3 paper_figures/fig11/make_fig11.py --source results
```

* SAGE (`greedy_basic.py`) and SAGE+ (`greedy.py`) on Marketing and CoverType (`covertype_expanded`), k = 5, θ = 0.1.
* x% of tuples means `--data_num x/100`.
* x% of attributes means `--max_columns round(x/100 · #columns)`, with 28 columns for Marketing and 110 for CoverType.
* Runtime covers every timed stage of the run.

---

## Notes

* `PROXY_DIR=<dir>` makes every run use the LLM-proxies in `<dir>` instead of `src/prompt_learning/model/`.
* `run_experiments.sh` settings can be overridden through the environment, e.g. `TRANSFORMER_PATH=~/hf_cache DATASETS="Video" bash paper_figures/table8/run_experiments.sh`. Logs go to `log/paper_*.txt`.
* `--no-caption` drops the caption text, so only the table or plot is rendered. `--out <dir>` changes the output folder.
