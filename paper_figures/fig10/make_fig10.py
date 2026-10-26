#!/usr/bin/env python3
"""Fig. 10: (left) online runtime of BruteForce and the PR/PX ablation; (right) utility vs. theta on Video.

    python paper_figures/fig10/make_fig10.py                    # values from the paper
    python paper_figures/fig10/make_fig10.py --source results   # recompute from result/

Right-panel paper values are digitized from the plot (about +/- 0.1).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Provenance, add_source_args, average_runs, compile_tex, find_runs, is_bruteforce, load_json, result_dirs  # noqa: E402
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = {"Marketing": "marketing_data_full", "Video": "Video", "House": "house"}
VARIANT_FLAGS = {
    "BF":     dict(do_prune=False, do_cache=False),
    "-PR&PX": dict(do_prune=False, do_cache=False),
    "-PX":    dict(do_prune=True, do_cache=False),
    "-PR":    dict(do_prune=False, do_cache=True),
    "SAGE":   dict(do_prune=True, do_cache=True),
}
# result/ sub-folders searched per variant (first match wins); run_experiments.sh writes the first one.
LEFT_DIRS = {
    "BF":     ["paper/fig10/bruteforce", "bf/ablation/all_off"],
    "-PR&PX": ["paper/fig10/ablation", "greedy/ablation/all_off", "new/greedy/ablation/all_off"],
    "-PX":    ["paper/fig10/ablation", "new/greedy/ablation/nocache"],
    "-PR":    ["paper/fig10/ablation", "greedy/ablation"],
    "SAGE":   ["paper/fig10/ablation", "greedy/ablation/noparallel"],
}
RIGHT_DIRS = ["paper/fig10/theta", "greedy/Video", "greedy/final/Video"]

BAR_STYLE = {
    "BF":     "draw=black, fill=white",
    "-PR&PX": "draw=black, fill=black!55",
    "-PX":    "draw=black, pattern=north east lines",
    "-PR":    "draw=black, densely dotted, pattern=dots, pattern color=black!70",
    "SAGE":   "draw=black, fill=black",
}
LINE_STYLE = {
    "5":  "color=blue!70, mark=*, mark options={fill=blue!60, fill opacity=0.7}",
    "10": "color=red!70, mark=square*, mark options={fill=red!60, fill opacity=0.7}",
    "15": "color=green!50!black, mark=triangle*, mark options={fill=green!60!black, fill opacity=0.7}",
    "20": "color=orange, mark=diamond*, mark options={fill=orange!80, fill opacity=0.7}",
}


def collect_results(paper, root, prov):
    left = {ds: list(v) for ds, v in paper["left"]["runtime"].items()}
    for ds, f in DATASET_FILE.items():
        for i, var in enumerate(paper["left"]["variants"]):
            got = None
            for rel in result_dirs(LEFT_DIRS[var]):
                base = os.path.join(root, rel)
                if not os.path.isdir(base):
                    continue
                dirs = find_runs(base, filename=f, k=4, threshold=0.1, max_columns=5, **VARIANT_FLAGS[var])
                dirs = [d for d in dirs if is_bruteforce(d) == is_bruteforce(rel)]
                if dirs:
                    got = average_runs(dirs)
                    break
            if got and got.get("T_online"):
                prov.add(f"left/{ds}/{var}", "results", "; ".join(got["dirs"]),
                         value=got["T_online"], paper=left[ds][i], metric="runtime", group=ds)
                left[ds][i] = got["T_online"]
            else:
                prov.add(f"left/{ds}/{var}", "paper")

    thetas = paper["right"]["theta"]
    right = {k: list(v) for k, v in paper["right"]["utility"].items()}
    for k in right:
        for j, th in enumerate(thetas):
            got = None
            for rel in result_dirs(RIGHT_DIRS):
                base = os.path.join(root, rel)
                if not os.path.isdir(base):
                    continue
                dirs = find_runs(base, filename="Video", k=int(k), threshold=th, data_num=1.0)  # all rows
                if dirs:
                    got = average_runs(dirs)
                    break
            if got and got.get("util") is not None:
                prov.add(f"right/k={k}/theta={th}", "results", "; ".join(got["dirs"]),
                         value=got["util"], paper=right[k][j], metric="utility", group="Video")
                right[k][j] = got["util"]
            else:
                prov.add(f"right/k={k}/theta={th}", "paper", "digitized")
    return left, right


def build_tex(paper, left, right, caption=True):
    variants = paper["left"]["variants"]
    datasets = paper["left"]["datasets"]
    bars = []
    for var in variants:
        i = variants.index(var)
        coords = " ".join(f"({ds},{max(left[ds][i], 1):g})" for ds in datasets)
        bars.append(f"\\addplot[{BAR_STYLE[var]}] coordinates {{{coords}}};")
    legend_left = ",".join(v.replace("&", "\\&") for v in variants)

    lines = []
    for k, ys in right.items():
        coords = " ".join(f"({th:g},{y:.3f})" for th, y in zip(paper["right"]["theta"], ys))
        lines.append(f"\\addplot[{LINE_STYLE.get(k, 'mark=*')}, thin] coordinates {{{coords}}};")
    legend_right = ",".join(f"$k$={k}" for k in right)
    caption_tex = (r"\par\vspace{4pt}\raggedright\small " + CAPTION) if caption else ""  # no backslash in {...} (Python 3.10)

    return rf"""\begin{{minipage}}{{15.5cm}}
\begin{{tikzpicture}}
\begin{{axis}}[
  name=left, width=7.2cm, height=4.1cm,
  ybar, bar width=5pt, ymode=log, log origin=infty, ymin=1, ymax=4e5,
  ytick={{10,100,1000,10000}}, ylabel={{Runtime (s)}}, ylabel style={{yshift=-2pt}},
  symbolic x coords={{{",".join(datasets)}}}, xtick=data, enlarge x limits=0.22,
  x tick label style={{yshift=-6pt}},
  nodes near coords={{\pgfmathprintnumber[fixed,precision=0,1000 sep={{}}]{{\pgfplotspointmeta}}}},
  point meta=rawy,
  every node near coord/.append style={{rotate=90, anchor=west, font=\scriptsize, text=black!55}},
  legend columns=5, legend style={{draw=none, at={{(0.5,1.03)}}, anchor=south, font=\small,
                                  /tikz/every even column/.append style={{column sep=4pt}}}},
  legend image code/.code={{\draw[#1] (0cm,-0.1cm) rectangle (0.28cm,0.1cm);}},
  tick label style={{font=\small}}, label style={{font=\small}},
]
{chr(10).join(bars)}
\legend{{{legend_left}}}
\end{{axis}}
\begin{{axis}}[
  at={{(left.east)}}, anchor=west, xshift=1.3cm, width=7.4cm, height=4.1cm,
  xmin=-0.07, xmax=1.1, ymin=0, ymax=11.5, xtick={{0,0.2,...,1.0}}, ytick={{0,2,...,10}},
  xticklabel style={{/pgf/number format/.cd, fixed, fixed zerofill, precision=1}},
  xlabel={{$\theta$}}, ylabel={{Utility}}, grid=major, grid style={{black!15}},
  legend columns=4, legend style={{draw=none, at={{(0.5,1.03)}}, anchor=south, font=\small,
                                  /tikz/every even column/.append style={{column sep=10pt}}}},
  tick label style={{font=\small}}, label style={{font=\small}}, mark size=2.2pt,
]
{chr(10).join(lines)}
\legend{{{legend_right}}}
\end{{axis}}
\end{{tikzpicture}}
{caption_tex}
\end{{minipage}}"""


CAPTION = (r"Fig. 10. (Left) Effect of optimization techniques. We used $k=4$ and $\theta=0.1$. "
           r"(Right) Effect of $\theta$ on \emph{Utility} on the Video dataset.")


def main():
    ap = add_source_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-caption", action="store_true")
    args = ap.parse_args()
    common.INCLUDE_OLD = args.include_old_results

    paper = load_json(os.path.join(HERE, "paper_values.json"))
    prov = Provenance("fig10")
    if args.source == "results":
        left, right = collect_results(paper, args.results, prov)
    else:
        left, right = paper["left"]["runtime"], paper["right"]["utility"]
        prov.add("all", "paper")
    compile_tex(build_tex(paper, left, right, caption=not args.no_caption), "fig10", args.out, args.keep_tex)
    prov.report(args.out)


if __name__ == "__main__":
    main()
