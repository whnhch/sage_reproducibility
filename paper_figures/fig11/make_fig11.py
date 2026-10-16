#!/usr/bin/env python3
"""Fig. 11: SAGE and SAGE+ runtime (every timed stage) vs. #tuples and #attributes on Marketing and CoverType.

    python paper_figures/fig11/make_fig11.py                    # values from the paper
    python paper_figures/fig11/make_fig11.py --source results   # recompute from result/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Provenance, add_source_args, average_runs, compile_tex, find_runs, load_json, result_dirs  # noqa: E402
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_COLUMNS = {"marketing_data_full": 28, "covertype_expanded": 110}
SERIES = {  # label -> (dataset file, variant)
    "SAGE (Marketing)":  ("marketing_data_full", "sage"),
    "SAGE+ (Marketing)": ("marketing_data_full", "sage_plus"),
    "SAGE (CoverType)":  ("covertype_expanded", "sage"),
    "SAGE+ (CoverType)": ("covertype_expanded", "sage_plus"),
}
# result/ sub-folders searched (first match wins); run_experiments.sh writes the first one.
DIRS = {
    ("tuples", "sage"):          ["paper/fig11/tuples/sage", "greedy/tuple"],
    ("tuples", "sage_plus"):     ["paper/fig11/tuples/sage_plus", "greedy/sampled/tuple"],
    ("attributes", "sage"):      ["paper/fig11/attributes/sage", "greedy/attribute"],
    ("attributes", "sage_plus"): ["paper/fig11/attributes/sage_plus", "greedy/sampled/attribute"],
}
STYLE = {
    "SAGE (Marketing)":  "draw=black, fill=green!60!black",
    "SAGE+ (Marketing)": "draw=black, fill=green!20, postaction={pattern=north east lines}",
    "SAGE (CoverType)":  "draw=black, fill=orange!80!red",
    "SAGE+ (CoverType)": "draw=black, fill=orange!30, postaction={pattern=north east lines}",
}
LEGEND = {
    "SAGE (Marketing)": r"SAGE (Marketing)", "SAGE+ (Marketing)": r"SAGE$^{+}$ (Marketing)",
    "SAGE (CoverType)": r"SAGE (CoverType)", "SAGE+ (CoverType)": r"SAGE$^{+}$ (CoverType)",
}


def collect_results(paper, root, prov):
    out = {panel: {s: list(v) for s, v in paper[panel].items()} for panel in ("tuples", "attributes")}
    for panel in out:
        for label, (f, variant) in SERIES.items():
            for i, pct in enumerate(paper["x"]):
                if panel == "tuples":
                    filt = dict(filename=f, k=5, data_num=pct / 100)
                else:
                    filt = dict(filename=f, k=5, data_num=1.0, max_columns=round(pct / 100 * N_COLUMNS[f]))
                got = None
                for rel in result_dirs(DIRS[(panel, variant)]):
                    base = os.path.join(root, rel)
                    if os.path.isdir(base):
                        dirs = find_runs(base, **filt)
                        if dirs:
                            got = average_runs(dirs)
                            break
                if got and got.get("T"):
                    prov.add(f"{panel}/{label}/{pct}%", "results", "; ".join(got["dirs"]),
                             value=got["T"], paper=out[panel][label][i], metric=f"runtime, {panel}",
                             group=label.split("(")[-1].rstrip(")"))
                    out[panel][label][i] = got["T"]
                else:
                    prov.add(f"{panel}/{label}/{pct}%", "paper")
    return out


def panel(name, data, xs, xlabel, extra=""):
    plots = []
    for label in SERIES:
        coords = " ".join(f"({x},{y:g})" for x, y in zip(xs, data[label]))
        plots.append(f"\\addplot[{STYLE[label]}] coordinates {{{coords}}};")
    return rf"""\begin{{axis}}[
  name={name}, {extra}width=7.6cm, height=4.0cm, ybar, bar width=5.4pt,
  ymin=0, ymax=600, clip=false, ytick={{0,100,...,500}}, ylabel={{Runtime (s)}}, xlabel={{{xlabel}}},
  xtick={{{",".join(map(str, xs))}}}, enlarge x limits=0.13, grid=major, grid style={{black!15}},
  nodes near coords, nodes near coords style={{rotate=90, anchor=west, font=\scriptsize, text=black!50,
                                              /pgf/number format/.cd, fixed, precision=0, 1000 sep={{}}}},
  tick label style={{font=\small}}, label style={{font=\small}},
]
{chr(10).join(plots)}
\end{{axis}}"""


def build_tex(paper, data, caption=True):
    xs = paper["x"]
    legend = " & ".join(
        rf"\tikz\draw[{STYLE[s]}] (0,0) rectangle (0.3,0.22); {LEGEND[s]}" for s in SERIES)
    caption_tex = (r"\par\vspace{4pt}\centering\small " + CAPTION) if caption else ""  # no backslash in {...} (Python 3.10)
    left = panel("left", data["tuples"], xs, "Tuples (\\%)")
    right = panel("right", data["attributes"], xs, "Attributes (\\%)", "at={(left.east)}, anchor=west, xshift=1.6cm, ")
    return rf"""\begin{{minipage}}{{16.5cm}}\centering
\begin{{tikzpicture}}
{left}
{right}
\end{{tikzpicture}}\\[2pt]
{{\small\setlength{{\tabcolsep}}{{10pt}}\begin{{tabular}}{{llll}}{legend}\end{{tabular}}}}
{caption_tex}
\end{{minipage}}"""


CAPTION = (r"Fig. 11. SAGE and SAGE$^{+}$ runtime (s) w.r.t (Left) \#tuples, (Right) \#attributes. "
           r"We used $k=5$ and $\theta=0.10$.")


def main():
    ap = add_source_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-caption", action="store_true")
    args = ap.parse_args()
    common.INCLUDE_OLD = args.include_old_results

    paper = load_json(os.path.join(HERE, "paper_values.json"))
    prov = Provenance("fig11")
    if args.source == "results":
        data = collect_results(paper, args.results, prov)
    else:
        data = {"tuples": paper["tuples"], "attributes": paper["attributes"]}
        prov.add("all", "paper")
    compile_tex(build_tex(paper, data, caption=not args.no_caption), "fig11", args.out, args.keep_tex)
    prov.report(args.out)


if __name__ == "__main__":
    main()
