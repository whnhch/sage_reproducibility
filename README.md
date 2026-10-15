# SAGE: Reproducibility Package

**Paper.** Whanhee Cho and Anna Fariha. *Data-Semantics-Aware Recommendation of Diverse Pivot Tables.* Proc. ACM Manag. Data 4(1) (SIGMOD), Article 23, February 2026. https://doi.org/10.1145/3786637

**Code.** https://github.com/whnhch/sage_reproducibility

This package installs SAGE, reruns the experiments of Section 6, and regenerates every table and figure next to the published version. Run all commands from the repository root.

---

## 1. Quick start

```bash
git clone https://github.com/whnhch/sage_reproducibility.git && cd sage_reproducibility
bash ARI/install.sh            # environment, large data files, models, checks, smoke test
bash ARI/run_experiments.sh    # Tables 8-9, Figs. 10-11 and the report
bash ARI/run_optional.sh --bf  # optional, GPU: BruteForce (also --llama, --gpu; Section 7)
```

The result is `ARI/out/reproduced_results.pdf`: each reproduced table and figure above the published one, then a plot of every reproduced value against the paper's. The individual PDFs are in `paper_figures/out/`. The default experiments need **no GPU and no Hugging Face token**.

---

## 2. What is reproduced

| Paper item | Content | Scripts | GPU |
|---|---|---|---|
| Table 8, §6.3 | SAGE, SAGE+, Top-k vs. DAISY, LLM, Excel, PowerBI, Google Sheets | `paper_figures/table8/` | no |
| Table 9, §6.4 | Pruning (PR) and LLM-proxy (PX) ablation vs. BruteForce | `paper_figures/table9/` | for BruteForce, no-proxy variants |
| Fig. 10, §6.4, §6.6 | Optimization runtimes; utility vs. $\theta$ | `paper_figures/fig10/` | for BruteForce, no-proxy variants |
| Fig. 11, §6.5 | Runtime vs. #tuples and #attributes | `paper_figures/fig11/` | no |
| §6.7 case studies | Recommended pivot tables | `selected_queries.json` of the Table 8 runs | no |

---

## 3. Requirements

**Paper's machine:** Ubuntu 22.04, 12 CPU cores, 256 GB RAM, NVIDIA H100 (96 GB), CUDA 12.8.

**Default experiments:** Linux, 64 GB RAM, 12 CPU cores (SAGE+ uses 12 worker processes), about 10 GB of disk. No GPU: the model outputs the runs need are shipped:

- Llama's attribute significance and aggregate ranking: `precompute/*.json`

- trend and surprise likelihoods: the LLM-proxy decision trees in `src/prompt_learning/model/`

- T5 / TAPEX embeddings of every candidate: `precompute/embeddings/`.

**Optional parts (`ARI/run_optional.sh`):** an NVIDIA GPU with ≥ 24 GB and 16 GB of disk for Llama.

**Software:** conda, and TeX Live with `pgfplots`, `libertine`, `newtx`, `standalone` and `pdfpages` to draw the figures.

**Models** (downloaded by `ARI/install.sh`):

| Model | Used for |
|---|---|
| `microsoft/tapex-large` | table (content) embedding (§4) |
| `gaussalgo/T5-LM-Large-text2sql-spider` | query embedding (§4) |
| `meta-llama/Meta-Llama-3-8B-Instruct` | LLM (§3); optional parts only, with `--llama`. Gated: accept Meta's license and put a token in `hf_token.txt` |

---

## 4. Setup

### 4.1 Get the system

Clone the repository as in Section 1.

`data/covertype_expanded.csv` (246 MB) is larger than GitHub allows; `install.sh` downloads it from Zenodo (DOI: TODO) and checks its SHA-256 (`ARI/scripts/fetch_data.py`).

### 4.2 Install

```bash
bash ARI/install.sh            # add --llama to also download Llama-3 for run_optional.sh
conda activate sage            # in new shells
```

`install.sh` builds a new conda environment `sage` from `ARI/env/environment.yml` (Python 3.10 and the pinned packages of `requirements.txt`), replacing an existing one of that name. It then downloads `covertype_expanded.csv`, T5 and TAPEX, and runs the checks of Section 4.4. The other scripts activate this environment and stop if a package version differs. To keep it elsewhere:

```bash
SAGE_ENV=/path/to/envs/sage bash ARI/install.sh      # and the same SAGE_ENV for the other scripts
```

### 4.3 Configure

| Setting | How | Default |
|---|---|---|
| Conda environment | `SAGE_ENV=name` or `SAGE_ENV=/path` | `sage` |
| Model cache | `TRANSFORMER_PATH=/path` | `~/.cache/huggingface` |
| Hugging Face token (optional parts only) | put it in the empty `hf_token.txt`, or `HF_TOKEN_PATH=/path` | `./hf_token.txt` |
| LLM-proxy models | `PROXY_DIR=...` | `src/prompt_learning/model` |

### 4.4 Check the installation

```bash
python ARI/scripts/check_install.py --online
bash ARI/scripts/smoke_test.sh
```

`check_install.py` reports OK / WARN / FAIL for packages, hardware, datasets, shipped inputs, models and LaTeX, and exits with 1 on any FAIL. `smoke_test.sh` runs SAGE once on Video ($k = 3$, $\theta = 0.3$) and ends with `SMOKE TEST PASSED` (2–5 min).

### 4.5 Data

| Paper name | File | Rows × cols | Source |
|---|---|---|---|
| Marketing | `data/marketing_data_full.csv` | 2,240 × 28 | https://www.kaggle.com/datasets/sahilnbajaj/marketing-campaigns-data-set |
| Video | `data/Video.csv` | 16,598 × 11 | https://www.kaggle.com/datasets/gregorut/videogamesales |
| House | `data/house.csv` | 1,460 × 81 | https://www.kaggle.com/datasets/animeshparikshya/house-sale-data81-column |
| CoverType | `data/covertype_expanded.csv` | 581,012 × 110 | https://archive.ics.uci.edu/ml/datasets/Covertype; its 55 attributes, each twice (`_1`, `_2`) |

Shipped inputs produced offline or collected by hand:

| Path | Content | Regenerate with |
|---|---|---|
| `src/prompt_learning/model/*_10000_*decision_tree.pkl` | LLM-proxies (§5.1.2) | shipped |
| `precompute/*_interesting_attrs.json`, `*_attribute_aggfunc_ranks.json` | Llama's attribute significance (§3.1.1) and aggregate ranking (§3.2.2) | delete them: runs then ask Llama |
| `precompute/embeddings/*.npz` | T5 / TAPEX vectors of every candidate | `python -m src.diversity.embedding_bank` (GPU) |
| `baselines/llm_*.csv`, `baselines/daisy_*.csv`, `daisy.pth` | LLM and DAISY recommendation lists; our DAISY classifier (§6.1.3) | `REGEN_BASELINES=1 bash paper_figures/table8/run_experiments.sh` |
| `baselines/{excel,powerbi,gsheets}_*.csv` | commercial tools' recommendation lists | by hand, below |

**Commercial tools.** Excel, PowerBI and Google Sheets are closed GUI tools. Each dataset was loaded into the tool and its pivot-table recommendations were recorded in order, one per row. Excel: Windows version 2501, built-in recommendations, February 2025. Google Sheets: browser version, suggested pivot tables, February 2025; at most 3 recommendations and none for House. PowerBI: "quick insights" in pivot-table form.

---

## 5. Experiments

**Setup.** `paper_figures/experiments_env.sh` holds the shared settings (seed 42, τ_ρ = 0.8, τ_π = 5.0, per-dataset γ, row fraction and column cap). Every run goes through `src/models/run_cached.py`, which calls the `src/models/*.py` scripts, loads Llama only if the run uses it, and reads the embeddings from `precompute/embeddings/`.

**Run.** Each `paper_figures/<item>/run_experiments.sh` loops over the paper's parameter grid. A run writes its results to `result/paper/<item>/<method>/<dataset>/` and a log to `log/paper_*.txt`; the recommended tables are in `diversity/.../selected_queries.json`. A failed run prints `FAILED, see log/...`.

**Resume.** A run whose results exist is skipped, so an interrupted call can be repeated; `FORCE=1` reruns it. To start over: `rm -rf result/paper/<item> log/paper_*`.

```bash
bash ARI/run_experiments.sh
bash ARI/run_experiments.sh --items fig11,table8
bash ARI/run_experiments.sh --items table8 --datasets Video
```

### 5.1 Table 8

* Marketing, Video, House; $k \in \lbrace 3, 5, 10 \rbrace$; $\theta = 0.30$ (Marketing, Video) or $0.20$ (House), and $0.10$ for $k = 10$.
* SAGE = `greedy_basic.py`; SAGE+ = `greedy.py` (samples max(50, 20%) of the pruned candidates, 12 workers); Top-k = `greedy_basic.py` with $\theta = 0$.
* Baselines: each list's k highest-utility tables, scored by `software.py`. Their T(s) was measured by hand and is taken from the paper.

### 5.2 Table 9 and Fig. 10

* Table 9: Marketing, first 5 attributes, $\theta = 0.1$, $k \in \lbrace 2, 5 \rbrace$. Fig. 10 left: the same variants, $k = 4$, on Marketing, Video and House. Fig. 10 right: SAGE on Video, $k \in \lbrace 5, 10, 15, 20 \rbrace$, $\theta \in \lbrace 0.0, \dots, 1.0 \rbrace$.
* Pruning threshold of the ablation: 0.5 for Marketing, 0.3 for Video and House.
* By default only the proxy variants run (No PR, SAGE). `bash ARI/run_optional.sh --bf` adds BruteForce and `--llama` adds No PR/PX and No PX. Skipped cells show the paper's values.

### 5.3 Fig. 11

SAGE and SAGE+ on Marketing and CoverType, $k = 5$, $\theta = 0.1$. x% of tuples = the first x% of rows; x% of attributes = the first round(x/100 · n) columns (n = 28 for Marketing, 110 for CoverType).

---

## 6. Tables, figures and report

Each `paper_figures/<item>/make_<item>.py` draws its table or figure in the paper's style to `paper_figures/out/<item>.pdf`: `--source results` from `result/paper/`, `--source paper` (default) from the paper's numbers in `paper_values.json`. `paper_figures/out/<item>_provenance.csv` records, per number, the run folders it came from or that it was taken from the paper. `ARI/scripts/make_report.py` combines everything into `ARI/out/reproduced_results.pdf`.

| Quantity | Computed from |
|---|---|
| Ins, Int, Util | sum over the k selected tables of `utility/utility_scores.csv`; Util = (Ins + Int) / 2 |
| m-dist, heatmap | `diversity/selected_distance_matrix.csv` |
| T(s), Fig. 11 runtime | pruning + utility + embedding + selection time |
| Online runtime (Table 9, Fig. 10) | utility + embedding + selection time |

```bash
bash ARI/run_experiments.sh --figures-only
```

---

## 7. Scripts

| Script | Does | GPU |
|---|---|---|
| `ARI/install.sh [--llama] [--no-smoke]` | environment, T5 + TAPEX (+ Llama), checks, smoke test | no |
| `ARI/run_experiments.sh [options]` | the default experiments, then the tables, figures and report | no |
| `ARI/run_optional.sh MODE [options]` | `--bf`: BruteForce (Table 9, Fig. 10 left; ~10 h). `--llama`: variants without the proxy, which call Llama (~5 h). `--gpu`: rerun with live T5 / TAPEX encoding | yes |

Options: `--items table8,table9,fig10,fig11`, `--datasets ...`, `--fig10-parts left|right`, `--iters 0,1,2` (repetitions to average; default one). `run_experiments.sh` also takes `--figures-only` (only draw) and `--paper-text` (τ_ρ = 0.5, τ_π = 2.0, pruning threshold 0.5; writes `result/paper_text/` and `ARI/out/reproduced_results_paper_text.pdf`). `--help` prints all of them.

```bash
bash ARI/run_experiments.sh --items fig11 --datasets covertype_expanded
bash ARI/run_optional.sh --llama --items table9
bash ARI/run_optional.sh --bf --llama --gpu
```

---

## 8. Repository map

```
sage_reproducibility/
├── README.md                     this file
├── ARI/                          reproducibility scripts
│   ├── install.sh                step 1: environment, models, checks
│   ├── run_experiments.sh        step 2: default experiments, figures, report
│   ├── run_optional.sh           optional GPU experiments (BruteForce, Llama, live encoding)
│   ├── env/                      environment.yml, requirements.txt
│   ├── scripts/                  common.sh, fetch_data.py, fetch_models.py, check_install.py, smoke_test.sh,
│   │                             make_report.py
│   └── out/                      reproduced_results.pdf
├── paper_figures/                one folder per table / figure
│   ├── run_all.sh, build_all.sh, experiments_env.sh, common.py
│   ├── table8/ table9/ fig10/ fig11/
│   │                             run_experiments.sh, make_<item>.py, paper_values.json
│   └── out/                      <item>.pdf, <item>_provenance.csv
├── src/
│   ├── models/                   greedy_basic.py (SAGE), greedy.py (SAGE+), bruteforce.py,
│   │                             software.py (scores a baseline list), llm.py, daisy.py,
│   │                             run_cached.py (runs them without unused models)
│   ├── utility/                  insightfulness.py (§3.1), interpretability.py (§3.2), utility.py (§3.3),
│   │                             llama.py, prompts/
│   ├── diversity/                helper.py (embeddings, distance, greedy selection; §4, §5.2),
│   │                             embedding_bank.py
│   ├── prompt_learning/model/    LLM-proxy decision trees (§5.1.2)
│   ├── helper.py                 pruning (§5.1.1), shared helpers
│   ├── generate_table.py         candidate pivot tables
│   └── preprocessing.py
├── precompute/
│   ├── <dataset>_*.json          Llama's attribute significance and aggregate ranking
│   └── embeddings/               T5 / TAPEX embedding banks
├── baselines/                    DAISY, LLM, Excel, PowerBI, Google Sheets recommendation lists
├── daisy.pth                     our DAISY classifier
├── data/                         marketing_data_full.csv, Video.csv, house.csv, covertype_expanded.csv
├── result/paper/                 run outputs (created by the experiments)
└── log/                          run logs
```
