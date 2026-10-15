#!/usr/bin/env python3
"""ARI/out/reproduced_results.pdf: each reproduced value against the paper's, then each table/figure
next to its published version. Reads <figures-dir>/<item>.pdf and <item>_provenance.csv.

    python ARI/scripts/make_report.py [--figures-dir DIR]
"""
import argparse
import csv
import datetime
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile

ARI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ARI)
FIG = os.path.join(ROOT, "paper_figures")
OUT = os.path.join(ARI, "out")
ITEMS = [("table8", "Table 8: SAGE vs. baselines (Section 6.3)"),
         ("table9", "Table 9: pruning / LLM-proxy ablation (Section 6.4)"),
         ("fig10", "Figure 10: optimization runtimes and effect of $\\theta$ (Sections 6.4, 6.6)"),
         ("fig11", "Figure 11: scalability (Section 6.5)")]

# Three colorblind-safe colors; each group also gets its own marker shape.
COLORS = ["2A78D6", "EB6834", "1BAF7A"]
MARKS = ["*", "square*", "triangle*"]
KEY_SHAPES = ["circle, minimum size=6pt", "rectangle, minimum size=5.5pt",
              "regular polygon, regular polygon sides=3, minimum size=8pt"]
METRIC_LABELS = {"T": "T (s)", "util": "Util", "ins": "Ins", "int": "Int", "mdist": "m-dist"}
SKIP_METRICS = {"n"}  # #PT is fixed by the method, not measured
TOLERANCE = 0.10


def read_provenance(fig_out, item):
    path = os.path.join(fig_out, f"{item}_provenance.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def comparable(rows):
    """(cell, metric, group, paper, reproduced) for every cell drawn from a run that has a paper value."""
    pairs = []
    for r in rows:
        if r["source"] != "results" or r.get("metric", "") in SKIP_METRICS:
            continue
        try:
            paper, mine = float(r.get("paper_value") or "x"), float(r.get("value") or "x")
        except ValueError:
            continue
        if math.isfinite(paper) and math.isfinite(mine):
            pairs.append((r["cell"], r.get("metric") or "value", r.get("group", ""), paper, mine))
    return pairs


def rel_diff(paper, mine):
    if abs(mine - paper) < 0.005:  # same value at the paper's two-decimal precision
        return 0.0
    if paper:
        return (mine - paper) / abs(paper)
    return 0.0 if mine == 0 else math.inf


def is_time(metric):
    return metric == "T" or "time" in metric or "runtime" in metric


def esc(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("&", "\\&")


def pct(x):
    return "inf" if math.isinf(x) else f"{x:+.0%}".replace("%", "\\%")


def panel_tex(metric, pairs, groups):
    """One square panel: reproduced (y) vs. paper (x), same scale on both axes."""
    log = is_time(metric) and all(p > 0 and m > 0 for _, _, _, p, m in pairs)
    vals = [v for _, _, _, p, m in pairs for v in (p, m)]
    lo, hi = min(vals), max(vals)
    if log:
        lo, hi = lo / 1.6, hi * 1.6
    else:
        pad = 0.08 * ((hi - lo) or abs(hi) or 1.0)
        lo, hi = (0.0 if lo >= 0 else lo - pad), hi + pad
    mode = "xmode=log, ymode=log, log ticks with fixed point, " if log else ""
    title = esc(METRIC_LABELS.get(metric, metric))
    L = [rf"\nextgroupplot[title={{{title}}}, {mode}xmin={lo:g}, xmax={hi:g}, ymin={lo:g}, ymax={hi:g}]"]
    band = (f"({lo:g},{lo * (1 - TOLERANCE):g}) ({hi:g},{hi * (1 - TOLERANCE):g}) "
            f"({hi:g},{hi * (1 + TOLERANCE):g}) ({lo:g},{lo * (1 + TOLERANCE):g})")
    L.append(rf"\addplot[fill=black!7, draw=none, forget plot] coordinates {{{band}}} -- cycle;")
    L.append(rf"\addplot[black!45, dashed, forget plot] coordinates {{({lo:g},{lo:g}) ({hi:g},{hi:g})}};")
    for gi, g in enumerate(groups):
        pts = " ".join(f"({p:g},{m:g})" for _, _, grp, p, m in pairs if grp == g)
        if pts:
            L.append(rf"\addplot[only marks, mark={MARKS[gi]}, mark size=2.4pt, color=s{gi + 1}, "
                     rf"mark options={{draw=white, line width=0.6pt}}, forget plot] coordinates {{{pts}}};")
    # ungrouped points, and groups past the third, in neutral ink
    rest = " ".join(f"({p:g},{m:g})" for _, _, grp, p, m in pairs if grp not in groups)
    if rest:
        L.append(rf"\addplot[only marks, mark=diamond*, mark size=2.6pt, color=black!60, "
                 rf"mark options={{draw=white, line width=0.6pt}}, forget plot] coordinates {{{rest}}};")
    return "\n".join(L)


def comparison_tex(pairs):
    if not pairs:
        return r"\textit{No values from runs yet. Run the experiments and figures stages first.}\\"
    metrics = list(dict.fromkeys(m for _, m, _, _, _ in pairs))
    named = list(dict.fromkeys(g for _, _, g, _, _ in pairs if g))
    groups = named[:len(COLORS)]  # fixed order of first appearance; never more than three hues
    cols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / cols)

    L = []
    if len(named) >= 2:  # a legend only when there are two or more groups
        keys = [rf"\tikz[baseline=-0.6ex]\node[draw=white, line width=0.6pt, fill=s{i + 1}, inner sep=0pt, "
                rf"{KEY_SHAPES[i]}]{{}};~{esc(g)}" for i, g in enumerate(groups)]
        if len(named) > len(groups):
            keys.append(r"\tikz[baseline=-0.6ex]\node[draw=white, fill=black!60, inner sep=0pt, minimum size=7pt, diamond]{};"
                        "~other (" + esc(", ".join(named[len(groups):])) + ")")
        L.append(r"\small " + r"\quad ".join(keys) + r"\\[4pt]")
    L.append(r"\begin{tikzpicture}\begin{groupplot}[group style={group size=%d by %d, horizontal sep=1.5cm, "
             r"vertical sep=1.7cm}, width=5.2cm, height=5.2cm, axis line style={black!40}, "
             r"tick style={black!40}, tick label style={font=\scriptsize, text=black!70}, "
             r"title style={font=\small}, xlabel={paper}, ylabel={reproduced}, "
             r"label style={font=\scriptsize, text=black!70}, grid=major, grid style={black!8}]" % (cols, nrows))
    for m in metrics:
        L.append(panel_tex(m, [p for p in pairs if p[1] == m], groups))
    L.append(r"\end{groupplot}\end{tikzpicture}\\[6pt]")

    diffs = [(abs(rel_diff(p, m)), c, p, m) for c, _, _, p, m in pairs]
    within = sum(d <= TOLERANCE + 1e-12 for d, *_ in diffs)
    med = sorted(d for d, *_ in diffs)[len(diffs) // 2]
    L.append(rf"\small {len(pairs)} values compared; {within} lie within $\pm${TOLERANCE:.0%} of the paper "
             rf"(shaded band); median $|\Delta|$ = {med:.0%}. Dashed line: reproduced = paper. "
             r"\\[8pt]".replace("%", r"\%"))
    L.append(r"\textbf{Largest differences}\\[2pt]\footnotesize"
             r"\begin{tabular}{lrrr}\toprule Cell & Paper & Reproduced & $\Delta$ \\ \midrule")
    for d, c, p, m in sorted(diffs, key=lambda x: x[0], reverse=True)[:8]:
        L.append(rf"\texttt{{{esc(c)}}} & {p:,.4g} & {m:,.4g} & {pct(rel_diff(p, m))} \\")
    L.append(r"\bottomrule\end{tabular}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-dir", default=os.path.join(FIG, "out"),
                    help="where make_<item>.py --source results wrote the PDFs and provenance CSVs")
    ap.add_argument("--output", default=os.path.join(OUT, "reproduced_results.pdf"), help="the report PDF")
    args = ap.parse_args()
    fig_out = args.figures_dir

    os.makedirs(os.path.join(OUT, "paper"), exist_ok=True)
    for item, _ in ITEMS:  # paper version, rendered the same way as the reproduced one
        subprocess.run([sys.executable, os.path.join(FIG, item, f"make_{item}.py"),
                        "--out", os.path.join(OUT, "paper")], check=True, stdout=subprocess.DEVNULL)

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "n/a"
    L = ["\\documentclass[10pt]{article}",
         "\\usepackage[margin=2cm]{geometry}\\usepackage[T1]{fontenc}\\usepackage{libertine}",
         "\\usepackage{graphicx,booktabs,xcolor,tikz,pgfplots}\\setlength{\\parindent}{0pt}",
         "\\usetikzlibrary{pgfplots.groupplots,shapes.geometric}\\pgfplotsset{compat=1.18}",
         "".join(f"\\definecolor{{s{i + 1}}}{{HTML}}{{{c}}}" for i, c in enumerate(COLORS)),
         "\\begin{document}",
         "{\\Large\\bfseries SAGE: reproduced results}\\\\[4pt]",
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {esc(platform.node())}, commit {esc(commit)}.\\\\",
         "Part 1 shows each reproduced table or figure above the paper's version. "
         "Part 2 plots, for each item, every value computed from a run against the value printed in the paper.",
         ("Parameters: $\\tau_\\rho = 0.5$, $\\tau_\\pi = 2.0$, pruning threshold 0.5.\\\\"
          if os.environ.get("PARAMS") == "paper_text" else "")]
    provs = {}
    for item, _ in ITEMS:
        rows = read_provenance(fig_out, item)
        provs[item] = (rows, (sum(r["source"] == "results" for r in rows), sum(r["source"] == "paper" for r in rows)) if rows else None)
    L.append("\\section*{Part 1: tables and figures}")
    for n, (item, title) in enumerate(ITEMS):
        mine, prov = os.path.join(fig_out, f"{item}.pdf"), provs[item][1]
        L.append(("\\clearpage" if n else "") + "\\subsection*{" + title + "}")
        if os.path.exists(mine) and prov:
            L.append("\\textbf{Reproduced}\\\\[4pt]")
            L.append(f"\\includegraphics[width=\\linewidth,height=0.36\\textheight,keepaspectratio]{{{mine}}}\\\\[10pt]")
        else:
            L.append("\\textbf{Reproduced:} not built yet (run the experiments stage).\\\\[10pt]")
        L.append("\\textbf{As printed in the paper}\\\\[4pt]")
        L.append(f"\\includegraphics[width=\\linewidth,height=0.36\\textheight,keepaspectratio]{{{os.path.join(OUT, 'paper', item + '.pdf')}}}")
    L.append("\\clearpage\\section*{Part 2: reproduced vs.\\ paper}")
    for n, (item, title) in enumerate(ITEMS):
        rows = provs[item][0]
        L.append(("\\clearpage" if n else "") + "\\subsection*{" + title + "}")
        if rows:
            L.append(comparison_tex(comparable(rows)))
        else:
            L.append("\\textit{Not built yet (run the figures stage).}\\\\")
    L.append("\\end{document}")

    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex not found")
    with tempfile.TemporaryDirectory() as tmp:
        tex = os.path.join(tmp, "reproduced_results.tex")
        with open(tex, "w") as f:
            f.write("\n".join(L))
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                               f"-output-directory={tmp}", tex], capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit("pdflatex failed:\n" + proc.stdout[-3000:])
        shutil.copy(os.path.join(tmp, "reproduced_results.pdf"), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
