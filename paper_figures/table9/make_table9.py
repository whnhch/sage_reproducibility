#!/usr/bin/env python3
"""Table 9 of sage.pdf: ablation of pruning (PR) and LLM-proxy (PX) vs. BruteForce
(Marketing, first 5 attributes, theta = 0.1, k = 2 and 5).

    python paper_figures/table9/make_table9.py                    # numbers as printed in the paper
    python paper_figures/table9/make_table9.py --source results   # recompute from result/

Variant -> run flags (see run_experiments.sh):
    BruteForce : src/models/bruteforce.py   --do_prune false --do_cache false
    No PR/PX   : src/models/greedy_basic.py --do_prune false --do_cache false
    No PR      : src/models/greedy_basic.py --do_prune false --do_cache true
    No PX      : src/models/greedy_basic.py --do_prune true  --do_cache false
    SAGE       : src/models/greedy_basic.py --do_prune true  --do_cache true
(--do_cache true means the LLM-proxy decision trees replace Llama calls.)
Utility / Ins / Int are reported as % of BruteForce; runtime is the online phase
(utility + embedding + selection; pruning is offline, Section 6.2).
"""
import argparse
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Provenance, add_source_args, average_runs, compile_tex, find_runs, is_bruteforce, load_json, result_dirs  # noqa: E402
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FILENAME = "marketing_data_full"
VARIANT_FLAGS = {
    "BruteForce": dict(do_prune=False, do_cache=False),
    "No PR/PX":   dict(do_prune=False, do_cache=False),
    "No PR":      dict(do_prune=False, do_cache=True),
    "No PX":      dict(do_prune=True, do_cache=False),
    "SAGE":       dict(do_prune=True, do_cache=True),
}
# Sub-folders of result/ that are searched for each variant (first hit wins).
VARIANT_DIRS = {
    "BruteForce": ["paper/table9/bruteforce", "bf/ablation/all_off"],
    "No PR/PX":   ["paper/table9/sage", "greedy/ablation/all_off", "new/greedy/ablation/all_off"],
    "No PR":      ["paper/table9/sage", "greedy/ablation"],
    "No PX":      ["paper/table9/sage", "new/greedy/ablation/nocache", "greedy/ablation"],
    "SAGE":       ["paper/table9/sage", "greedy/ablation/noparallel", "greedy/ablation"],
}


def collect_results(paper, results_root, prov):
    """Replace paper numbers by result/ numbers wherever a matching run exists."""
    out = {kk: {key: dict(v) for key, v in blk.items()} for kk, blk in paper["blocks"].items()}
    n_pts = dict(paper["n_pts"])
    for kk in paper["blocks"]:
        k = int(kk)
        runs = {}
        for var, flags in VARIANT_FLAGS.items():
            for rel in result_dirs(VARIANT_DIRS[var]):
                base = os.path.join(results_root, rel)
                if not os.path.isdir(base):
                    continue
                dirs = find_runs(base, filename=FILENAME, k=k, threshold=0.1, max_columns=5, **flags)
                if var == "BruteForce":
                    dirs = [d for d in dirs if is_bruteforce(d)]
                elif is_bruteforce(base):
                    continue
                if dirs:
                    runs[var] = average_runs(dirs)
                    break
        bf = runs.get("BruteForce")
        for var in paper["approaches"]:
            r = runs.get(var)
            if r is None:
                for key in ("runtime", "utility", "ins", "int"):
                    prov.add(f"k={k}/{var}/{key}", "paper")
                continue
            out[kk]["runtime"][var] = r["T_online"]
            prov.add(f"k={k}/{var}/runtime", "results", "; ".join(r["dirs"]),
                     value=r["T_online"], paper=paper["blocks"][kk]["runtime"][var], metric="runtime",
                     group="Marketing")
            if r.get("n_candidates"):
                n_pts[var] = r["n_candidates"]
            if var == "BruteForce" and r.get("n_candidates"):
                out[kk]["combinations"][var] = human_count(comb(r["n_candidates"], k))
            for key in ("utility", "ins", "int"):
                src_key = "util" if key == "utility" else key
                if bf and bf.get(src_key) and r.get(src_key) is not None:
                    out[kk][key][var] = 100.0 * r[src_key] / bf[src_key]
                    # BruteForce is 100% by definition, so it says nothing about agreement
                    prov.add(f"k={k}/{var}/{key}", "results", "relative to BruteForce",
                             value=None if var == "BruteForce" else out[kk][key][var],
                             paper=paper["blocks"][kk][key][var], metric=f"{key} (% of BruteForce)",
                             group="Marketing")
                else:
                    prov.add(f"k={k}/{var}/{key}", "paper", "no BruteForce run to normalise against")
    return out, n_pts


def human_count(n):
    if n >= 1_000_000:
        return f"{round(n / 1_000_000)}M"
    return f"{n:,}"


def num(v):
    if isinstance(v, str):
        return v
    return f"{int(round(v)):,}"


def build_tex(paper, blocks, n_pts, caption=True):
    apps = paper["approaches"]
    head = " & ".join(f"\\textbf{{{a}}}" for a in apps)
    L = []
    L.append("\\begin{minipage}{12.6cm}\\centering\\small")
    L.append("\\begin{tabular}{@{}cl" + "r" * len(apps) + "@{}}")
    L.append("\\toprule")
    L.append(f" & \\textbf{{Approach}} & {head} \\\\")
    L.append("\\midrule")
    L.append(" & \\#Pivot Tables (PTs) & " + " & ".join(num(n_pts[a]) for a in apps) + " \\\\")
    rows = [("combinations", "\\#PT combinations"), ("runtime", "Runtime (s)"), ("utility", "Utility (\\%)"),
            ("ins", "-- Insightfulness (\\%)"), ("int", "-- Interpretability (\\%)")]
    for kk in sorted(blocks, key=int):
        L.append("\\midrule")
        for i, (key, label) in enumerate(rows):
            lead = (f"\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{K = {kk}}}}}}}" if i == 0 else "")
            L.append(f"{lead} & {label} & " + " & ".join(num(blocks[kk][key][a]) for a in apps) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\par\\vspace{4pt}\\footnotesize")
    L.append("\\fbox{\\begin{tabular}{@{}p{5.2cm}p{6.4cm}@{}}"
             "PR = Pruning optimization & No PR/PX = Plain greedy, no optimization \\\\"
             "PX = LLM-Proxy optimization & SAGE = Greedy + PR + PX"
             "\\end{tabular}}")
    if caption:
        L.append("\\par\\vspace{8pt}\\raggedright\\small " + CAPTION)
    L.append("\\end{minipage}")
    return "\n".join(L)


CAPTION = (r"Table 9. Comparison among variants for $\theta=0.1$ on the Marketing dataset over 5 attributes. We "
           r"report the runtimes for the online phase for SAGE and other variants when optimizations are applied.")


def main():
    ap = add_source_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-caption", action="store_true")
    args = ap.parse_args()
    common.INCLUDE_OLD = args.include_old_results

    paper = load_json(os.path.join(HERE, "paper_values.json"))
    prov = Provenance("table9")
    if args.source == "results":
        blocks, n_pts = collect_results(paper, args.results, prov)
    else:
        blocks, n_pts = paper["blocks"], paper["n_pts"]
        prov.add("all", "paper")
    compile_tex(build_tex(paper, blocks, n_pts, caption=not args.no_caption), "table9", args.out, args.keep_tex)
    prov.report(args.out)


if __name__ == "__main__":
    main()
