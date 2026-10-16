"""Shared helpers for Tables 8-9 and Figs. 10-11: read run folders into the paper's metrics, compile LaTeX.
Standard library only."""
import ast
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root (folder that holds src/, result/, sage.pdf)
DEFAULT_RESULTS = os.path.join(ROOT, "result")
DEFAULT_OUT = os.path.join(HERE, "out")

# Folder names come from src/helper.py::param_dict_to_folder:
#   "_".join(f"{k}{str(v).replace('.', '-')}" for k, v in sorted(params.items()))
_PARAM_KEYS = ["data_num", "do_cache", "do_parallel", "do_prune", "filename",
               "int_threshold", "k", "max_columns", "threshold"]


# --------------------------------------------------------------------------- #
# Reading run folders
# --------------------------------------------------------------------------- #
def parse_run_folder(name):
    """'data_num1-0_..._k5_max_columns90_threshold0-1' -> {'k': 5, 'threshold': 0.1, ...}"""
    params = {}
    keys_by_len = sorted(_PARAM_KEYS, key=len, reverse=True)
    # split at "_<key>" boundaries; "threshold" must not split "int_threshold"
    tokens = re.split(r"_(?=(?:data_num|do_cache|do_parallel|do_prune|filename|int_threshold"
                      r"|k(?=\d)|max_columns|(?<!int_)threshold))", name)
    for tok in tokens:
        key = next((k for k in keys_by_len if tok.startswith(k)), None)
        if key is None:
            continue
        raw = tok[len(key):]
        if raw in ("True", "False"):
            params[key] = raw == "True"
        elif re.fullmatch(r"\d+-\d+", raw):
            params[key] = float(raw.replace("-", "."))
        elif re.fullmatch(r"\d+", raw):
            params[key] = int(raw)
        else:
            params[key] = raw
    return params


def find_runs(base_dir, **filters):
    """All diversity run folders under base_dir whose parsed params match filters.

    Floats are compared with a tolerance; a filter value of None is ignored.
    """
    hits = []
    pattern = os.path.join(base_dir, "**", "diversity", "data_num*")
    for d in sorted(glob.glob(pattern, recursive=True)):
        name = os.path.basename(d)
        if name.startswith("._"):
            continue
        if not os.path.exists(os.path.join(d, "diversity_result.csv")):
            continue  # unfinished run (stopped or crashed before selection): only params.json was written
        p = parse_run_folder(name)
        ok = True
        for key, want in filters.items():
            if want is None:
                continue
            got = p.get(key)
            if isinstance(want, float) or isinstance(got, float):
                ok = got is not None and abs(float(got) - float(want)) < 1e-9
            else:
                ok = got == want
            if not ok:
                break
        if ok:
            hits.append(d)
    return hits


def _read_kv_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return {r[0]: r[1] for r in rows[1:] if len(r) >= 2}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _timing(path):
    """Seconds in a timing CSV: either one row of *_sec columns (greedy) or Metric/Value rows (bruteforce)."""
    if not os.path.exists(path):
        return 0.0
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if not rows:
        return 0.0
    if rows[0][0] == "Metric":
        kv = {r[0]: r[1] for r in rows[1:] if len(r) >= 2}
        return _float(kv.get("Elapsed time (seconds)"))
    return sum(_float(v) for v in rows[1]) if len(rows) > 1 else 0.0


def _utility_key(index, column, value, aggfunc):
    """Same key that src/utility/helper.py writes: (index, [col] or '', value, AGG)."""
    if isinstance(column, str) and column not in ("", "[]"):
        column = [column]
    elif not isinstance(column, list) or not column:
        column = ""
    if isinstance(value, list):
        value = tuple(value)
    return (index, tuple(column) if column else "", value, str(aggfunc).upper())


def _load_utility_scores(path):
    scores = {}
    if not os.path.exists(path):
        return scores
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx, col, val, agg = ast.literal_eval(row["key"])
            except (ValueError, SyntaxError):
                continue
            scores[_utility_key(idx, col, val, agg)] = row
    return scores


def read_matrix(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return [[0.0 if c.strip() == "inf" else float(c) for c in r] for r in rows]


def run_metrics(div_dir):
    """Metrics of one run. T = every timed stage; T_online drops pruning (offline, Section 6.2).
    Util / Ins / Int are sums over the selected pivot tables."""
    name = os.path.basename(div_dir)
    stage_root = os.path.dirname(os.path.dirname(div_dir))
    prune_dir = os.path.join(stage_root, "prune", name)
    util_dir = os.path.join(stage_root, "utility", name)

    t_prune = _float(_read_kv_csv(os.path.join(prune_dir, "pruned_time.csv")).get("Elapsed time (seconds)"))
    # parallel path writes "Total elapsed time", the sequential one "Total query elapsed time"
    util_time = _read_kv_csv(os.path.join(util_dir, "utility_total_time.csv"))
    t_util = _float(util_time.get("Total elapsed time (seconds)", util_time.get("Total query elapsed time (seconds)")))
    scores = _load_utility_scores(os.path.join(util_dir, "utility_scores.csv"))
    n_candidates = int(util_time["Computed combinations"]) if util_time.get("Computed combinations") else (len(scores) or None)
    t_emb = _timing(os.path.join(div_dir, "embedding_time.csv"))
    t_sel = _timing(os.path.join(div_dir, "diversity_time.csv"))

    out = dict(params=parse_run_folder(name), dir=div_dir,
               t_prune=t_prune, t_util=t_util, t_emb=t_emb, t_sel=t_sel,
               T=t_prune + t_util + t_emb + t_sel, T_online=t_util + t_emb + t_sel,
               n_candidates=n_candidates,
               n=None, util=None, ins=None, int=None, mdist=None, matrix=None)

    res_path = os.path.join(div_dir, "diversity_result.csv")
    if not os.path.exists(res_path):
        return out
    with open(res_path, newline="") as f:
        sel_rows = list(csv.DictReader(f))
    out["n"] = len(sel_rows)
    out["util"] = sum(float(r["utility_score"]) for r in sel_rows)
    dists = [float(r["min_distance_to_others"]) for r in sel_rows if r["min_distance_to_others"] != "inf"]
    out["mdist"] = min(dists) if dists else None

    ins = itp = 0.0
    matched = 0
    for r in sel_rows:
        q = ast.literal_eval(r["query"])
        row = scores.get(_utility_key(q["index"], q["column"], q["value"], q["aggfunc"]))
        if row:
            ins += float(row["insightfulness score"])
            itp += float(row["interpretability score"])
            matched += 1
    if matched == len(sel_rows) and sel_rows:
        out["ins"], out["int"] = ins, itp

    out["matrix"] = read_matrix(os.path.join(div_dir, "selected_distance_matrix.csv"))
    return out


def average_runs(div_dirs):
    """Average run_metrics over repeated iterations."""
    runs = [run_metrics(d) for d in div_dirs]
    if not runs:
        return None
    avg = {"iters": len(runs), "dirs": div_dirs, "matrix": runs[0]["matrix"]}
    for key in ("n", "n_candidates", "T", "T_online", "t_prune", "t_util", "t_emb", "t_sel", "ins", "int", "util", "mdist"):
        vals = [r[key] for r in runs if r[key] is not None]
        avg[key] = mean(vals) if vals else None
    if avg["n"] is not None:
        avg["n"] = round(avg["n"])
    return avg


# --------------------------------------------------------------------------- #
# Paper values / provenance
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path) as f:
        return json.load(f)


class Provenance:
    """Keeps track of which numbers came from result/ and which from the paper."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, cell, source, detail="", value=None, paper=None, metric="", group=""):
        """value / paper: drawn and printed numbers of this cell; metric and group set make_report.py's panel
        and color."""
        self.rows.append((cell, source, detail, "" if value is None else value,
                          "" if paper is None else paper, metric, group))

    def report(self, out_dir):
        path = os.path.join(out_dir, f"{self.name}_provenance.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "source", "detail", "value", "paper_value", "metric", "group"])
            w.writerows(self.rows)
        n_res = sum(1 for r in self.rows if r[1] == "results")
        n_pap = sum(1 for r in self.rows if r[1] == "paper")
        print(f"[{self.name}] {n_res} values from result/, {n_pap} from paper_values.json -> {path}")


INCLUDE_OLD = False  # set by --include-old-results


def is_bruteforce(path):
    """True for a BruteForce run folder: paper/.../bruteforce/... (or bf/ in the older result folders)."""
    return bool({"bruteforce", "bf"} & set(path.replace("\\", "/").split("/")))


def result_dirs(rels):
    """Run folders to search. By default only those written by paper_figures/*/run_experiments.sh (paper/...);
    with --include-old-results also older result folders."""
    return [r for r in rels if INCLUDE_OLD or r.startswith("paper/")]


def add_source_args(parser):
    parser.add_argument("--source", choices=["paper", "results"], default="paper",
                        help="paper: render the numbers reported in sage.pdf (paper_values.json). "
                             "results: recompute from run folders under --results; values that "
                             "cannot be found fall back to the paper and are listed in *_provenance.csv")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="root of the result/ tree")
    parser.add_argument("--include-old-results", action="store_true",
                        help="also use older run folders (e.g. result/greedy/final) where result/paper/ has no run")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where .tex/.pdf are written")
    parser.add_argument("--keep-tex", action="store_true", help="keep LaTeX aux files for debugging")
    return parser


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{booktabs,multirow,graphicx,xcolor,array,amsmath}
\usepackage{tikz,pgfplots}
\usetikzlibrary{patterns,pgfplots.groupplots}
\pgfplotsset{compat=1.18}
"""


def fmt(x, nd=2, dash="--"):
    if x is None:
        return dash
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def compile_tex(body, name, out_dir, keep=False):
    """Write <out_dir>/<name>.tex and compile it to <out_dir>/<name>.pdf."""
    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f"{name}.tex")
    with open(tex_path, "w") as f:
        f.write(PREAMBLE + "\\begin{document}\n" + body + "\n\\end{document}\n")
    if shutil.which("pdflatex") is None:
        sys.exit(f"pdflatex not found; LaTeX source written to {tex_path}")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               f"-output-directory={tmp}", tex_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log = proc.stdout[-3000:]
            sys.exit(f"pdflatex failed for {tex_path}:\n{log}")
        shutil.copy(os.path.join(tmp, f"{name}.pdf"), os.path.join(out_dir, f"{name}.pdf"))
        if keep:
            shutil.copy(os.path.join(tmp, f"{name}.log"), os.path.join(out_dir, f"{name}.log"))
    print(f"[{name}] wrote {os.path.join(out_dir, name + '.pdf')}")
