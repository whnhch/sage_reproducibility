#!/usr/bin/env python3
"""Table 8 of sage.pdf: SAGE / SAGE+ vs. baselines on Marketing, Video, House for k = 3, 5, 10.

    python paper_figures/table8/make_table8.py                    # numbers as printed in the paper
    python paper_figures/table8/make_table8.py --source results   # recompute from result/

Raw results come from paper_figures/table8/run_experiments.sh (wraps src/models/greedy*.py,
llm.py, daisy.py and software.py). Bold = best Util / m-dist in a (k, dataset) block,
underline = second best.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (Provenance, add_source_args, average_runs, compile_tex, find_runs, fmt,  # noqa: E402
                    load_json, read_matrix, result_dirs)
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = {"Marketing": "marketing_data_full", "Video": "Video", "House": "house"}
METHODS = ["Top-k", "DAISY", "LLMs", "PowerBI", "GSheets", "Excel", "SAGE", "SAGE+"]
COLS = ["n", "T", "ins", "int", "util", "mdist"]

# Where each method's runs live under result/ ({f} = dataset file name).
#   kind "greedy":   folders written by greedy.py / greedy_basic.py (diversity_result.csv)
#   kind "software": folders written by software.py (utility_scores.csv + combined_sqrt_cosine_distance.csv);
#                    each list's k highest-utility tables are scored.
RESULT_LOCATIONS = {
    "SAGE":    {"kind": "greedy", "dirs": ["paper/table8/sage/{f}", "greedy/final/{f}"]},
    "SAGE+":   {"kind": "greedy", "dirs": ["paper/table8/sage_plus/{f}", "greedy/sampled/{f}"]},
    # Top-k = the same pipeline with no diversity constraint (theta = 0)
    "Top-k":   {"kind": "greedy", "dirs": ["paper/table8/topk/{f}", "greedy/sampled/{f}"], "theta": 0.0},
    "DAISY":   {"kind": "software", "dirs": ["paper/table8/daisy", "daisy/result"]},
    "LLMs":    {"kind": "software", "dirs": ["paper/table8/llm_k{k}"]},
    "PowerBI": {"kind": "software", "dirs": ["paper/table8/powerbi", "powerbi/result"]},
    "GSheets": {"kind": "software", "dirs": ["paper/table8/gsheets", "gsheets/result"]},
    "Excel":   {"kind": "software", "dirs": ["paper/table8/excel", "excel/result"]},
}


# --------------------------------------------------------------------------- #
# results mode
# --------------------------------------------------------------------------- #
def greedy_cell(results_root, loc, dataset, k, theta):
    f = DATASET_FILE[dataset]
    for rel in result_dirs(loc["dirs"]):
        base = os.path.join(results_root, rel.format(f=f))
        if not os.path.isdir(base):
            continue
        filt = dict(filename=f, k=k, threshold=loc.get("theta", theta))
        dirs = find_runs(base, **filt)
        if dirs:
            return average_runs(dirs)
    return None


def software_cell(results_root, loc, dataset, k):
    """Score the k highest-utility pivot tables a tool recommended (software.py output).

    software.py writes <out>/utility/<params>/utility_scores.csv and
    <out>/diversity/<params>/combined_sqrt_cosine_distance.csv; T(s) is not
    measured there (tools were run by hand), so it always comes from the paper.
    """
    f = DATASET_FILE[dataset]
    for rel in result_dirs(loc["dirs"]):
        util_root = os.path.join(results_root, rel.format(k=k), "utility")
        if not os.path.isdir(util_root):
            continue
        names = [n for n in sorted(os.listdir(util_root)) if f"filename{f}_" in n]
        names = [n for n in names if f"_k{k}_" in n] or names
        for name in names:
            util_path = os.path.join(util_root, name, "utility_scores.csv")
            if not os.path.exists(util_path):
                continue
            with open(util_path, newline="") as fh:
                all_rows = list(csv.DictReader(fh))
            idx = sorted(sorted(range(len(all_rows)), key=lambda i: -float(all_rows[i]["utility score"]))[:k])
            rows = [all_rows[i] for i in idx]
            if not rows:
                continue
            full = read_matrix(os.path.join(results_root, rel.format(k=k), "diversity", name,
                                            "combined_sqrt_cosine_distance.csv"))
            matrix = mdist = None
            if full:
                if len(full) == len(full[0]) + 1:  # header row written by DataFrame.to_csv
                    full = full[1:]
                if len(full) != len(all_rows):
                    sys.exit(f"{name}: {len(full)}x{len(full)} distance matrix for {len(all_rows)} tables")
                n = len(idx)
                matrix = [[full[i][j] for j in idx] for i in idx]
                off = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
                mdist = min(off) if off else None
            return dict(n=len(rows), T=None,
                        ins=sum(float(r["insightfulness score"]) for r in rows),
                        int=sum(float(r["interpretability score"]) for r in rows),
                        util=sum(float(r["utility score"]) for r in rows),
                        mdist=mdist, matrix=matrix, iters=1)
    return None


def collect_results(paper, results_root, prov):
    """Fill every cell from result/ where possible, else from the paper."""
    table = {}
    for block in paper["blocks"]:
        k = block["k"]
        for method, per_ds in block["rows"].items():
            loc = RESULT_LOCATIONS.get(method)
            for ds in paper["datasets"]:
                paper_row = per_ds.get(ds)
                theta = block["theta"][ds]
                got = None
                if loc and paper_row is not None:
                    if loc["kind"] == "greedy":
                        got = greedy_cell(results_root, loc, ds, k, theta)
                    else:
                        got = software_cell(results_root, loc, ds, k)
                table[(k, method, ds)] = merge_row(paper_row, got, f"k={k}/{method}/{ds}", prov, ds)
    return table


def merge_row(paper_row, got, cell, prov, group=""):
    """Per column: the result/ value if there is one, else the paper value."""
    if paper_row is None:
        return None, None
    row = []
    for i, col in enumerate(COLS):
        val = got.get(col) if got else None
        if val is None:
            row.append(paper_row[i])
            prov.add(f"{cell}/{col}", "paper")
        else:
            row.append(val)
            prov.add(f"{cell}/{col}", "results", "; ".join(got.get("dirs", [])) or f"{got.get('iters', 1)} run(s)",
                     value=val, paper=paper_row[i], metric=col, group=group)
    return row, (got or {}).get("matrix")


def collect_paper(paper, prov):
    table = {}
    for block in paper["blocks"]:
        for method, per_ds in block["rows"].items():
            for ds in paper["datasets"]:
                table[(block["k"], method, ds)] = (per_ds.get(ds), None)
                prov.add(f"k={block['k']}/{method}/{ds}", "paper")
    return table


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def heatmap(matrix, size_mm=3.4, dmax=0.6):
    """k x k distance matrix as a tiny grayscale grid: 0 (self) = black, far = white."""
    if not matrix:
        return ""
    n = len(matrix)
    c = size_mm / n
    cmds = []
    for i in range(n):
        for j in range(n):
            d = 0.0 if i == j else max(0.0, matrix[i][j])
            black = int(round(100 * (1 - min(d / dmax, 1.0))))
            cmds.append(f"\\fill[black!{black}] ({j * c:.3f}mm,{-(i + 1) * c:.3f}mm) rectangle +({c:.3f}mm,{c:.3f}mm);")
    return ("\\raisebox{-0.9mm}{\\begin{tikzpicture}" + "".join(cmds) +
            f"\\draw[black!60,line width=0.2pt] (0,0) rectangle ({size_mm}mm,{-size_mm}mm);\\end{{tikzpicture}}}}")


def rank_marks(values):
    """Indices of best (bold) and second best (underline) values; ties share the mark."""
    present = sorted({round(v, 2) for v in values if v is not None}, reverse=True)
    best = present[0] if present else None
    second = present[1] if len(present) > 1 else None
    marks = []
    for v in values:
        if v is None:
            marks.append(None)
        elif round(v, 2) == best:
            marks.append("b")
        elif round(v, 2) == second:
            marks.append("u")
        else:
            marks.append(None)
    return marks


def styled(text, mark):
    if mark == "b":
        return f"\\textbf{{{text}}}"
    if mark == "u":
        return f"\\underline{{{text}}}"
    return text


def method_label(m):
    if m == "SAGE":
        return "\\textbf{SAGE}"
    if m == "SAGE+":
        return "\\textbf{SAGE}$^{+}$"
    return m


def build_tex(paper, table, caption=True):
    ds_list = paper["datasets"]
    lines = []
    lines.append("\\begin{minipage}{\\linewidth}\\centering\\footnotesize\\setlength{\\tabcolsep}{2.6pt}")
    lines.append("\\begin{tabular}{@{}c|l|" + "|".join(["rrrrrrc"] * len(ds_list)) + "@{}}")
    lines.append("\\toprule")
    lines.append(" & & " + " & ".join(f"\\multicolumn{{7}}{{c{'|' if i < len(ds_list) - 1 else ''}}}{{\\textbf{{{d}}}}}"
                                      for i, d in enumerate(ds_list)) + " \\\\")
    sub = ("\\#PT & T(s) & Ins & Int & Util & \\shortstack{Div\\\\[-1pt]{\\tiny m-dist}} & "
           "\\shortstack{Div\\\\[-1pt]{\\tiny heatmap}}")
    lines.append(" & & " + " & ".join([sub] * len(ds_list)) + " \\\\")
    lines.append("\\midrule")
    for bi, block in enumerate(paper["blocks"]):
        k = block["k"]
        methods = [m for m in METHODS if m in block["rows"]]
        marks = {}
        for ds in ds_list:
            rows = [table[(k, m, ds)][0] for m in methods]  # (row, matrix)
            marks[(ds, "util")] = rank_marks([r[4] if r else None for r in rows])
            marks[(ds, "mdist")] = rank_marks([r[5] if r else None for r in rows])
        for mi, m in enumerate(methods):
            first = (f"\\multirow{{{len(methods)}}}{{*}}{{\\rotatebox{{90}}{{\\scriptsize {block['label']}}}}}"
                     if mi == 0 else "")
            cells = [first, method_label(m)]
            for ds in ds_list:
                row, matrix = table[(k, m, ds)]
                if row is None:
                    dash = "--" if m == "GSheets" else "-"
                    cells += [dash] * 7
                    continue
                n, t, ins, itp, util, md = row
                vals = [str(int(round(n))), str(int(round(t))), fmt(ins), fmt(itp),
                        styled(fmt(util), marks[(ds, "util")][mi]),
                        styled(fmt(md), marks[(ds, "mdist")][mi])]
                cells += vals + [heatmap(matrix)]
            lines.append(" & ".join(cells) + " \\\\[1.2pt]")
        lines.append("\\bottomrule" if bi == len(paper["blocks"]) - 1 else "\\midrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append("\\par\\vspace{6pt}\\raggedright\\small\\setlength{\\parindent}{0pt}")
        lines.append(CAPTION)
    lines.append("\\end{minipage}")
    width = "17.5cm"
    return f"\\begin{{minipage}}{{{width}}}\n" + "\n".join(lines) + "\n\\end{minipage}"


CAPTION = (r"Table 8. Comparison with baselines across three datasets and three values of $k$. For $k=3$ and $k=5$, "
           r"we used $\theta=0.30$ on Marketing and Video, and $\theta=0.20$ on House. For $k=10$, $\theta=0.10$ was "
           r"used across all datasets. Columns show the number of pivot tables recommended (\#PT), elapsed time "
           r"(T, in seconds), \emph{Insightfulness} (Ins), \emph{Interpretability} (Int), \emph{Utility} (Util), and "
           r"\emph{Diversity} (Div) in terms of minimum pairwise distance (m-dist) and a distance matrix visualization "
           r"(heatmap). In the distance matrix, the diagonal is black, denoting 0 self distance from a pivot table to "
           r"itself. Lighter colors denote less similar pairs, offering diversity. For Excel, PowerBI, and Google "
           r"Sheets, $k$ cannot be controlled. Thus, we consider the first-$k$ items when more than $k$ are returned. "
           r"Google Sheets fails to produce more than 3 recommendations, thus, we exclude it when $k>3$. The best "
           r"values in Util and m-dist are marked as bold and the second best underlined.")


def main():
    ap = add_source_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-caption", action="store_true")
    args = ap.parse_args()
    common.INCLUDE_OLD = args.include_old_results

    paper = load_json(os.path.join(HERE, "paper_values.json"))
    prov = Provenance("table8")
    if args.source == "results":
        table = collect_results(paper, args.results, prov)
    else:
        table = collect_paper(paper, prov)
    body = build_tex(paper, table, caption=not args.no_caption)
    compile_tex(body, "table8", args.out, args.keep_tex)
    prov.report(args.out)


if __name__ == "__main__":
    main()
