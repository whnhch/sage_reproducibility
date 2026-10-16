#!/usr/bin/env python3
"""Check that this machine can run the experiments; prints OK / WARN / FAIL, exits 1 on any FAIL.

    python ARI/scripts/check_install.py [--online] [--render-only] [--skip-hash]
"""
import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys

ARI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ARI)

MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/tapex-large", "gaussalgo/T5-LM-Large-text2sql-spider"]
DATASETS = {  # file: (rows, columns, sha256)
    "marketing_data_full.csv": (2240, 28, "6005c678856c0bc249c582a993b91ea887305f9618078caa3d59ee045d5ae7aa"),
    "Video.csv": (16598, 11, "e2076095ffcae2a92dbc6de6ecbd54455ee034bb38da550b2dc385e9265d4ebe"),
    "house.csv": (1460, 81, "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"),
    "covertype_expanded.csv": (581012, 110, "ab286fca6639da51e7f2659e3faa0015096543ab0ff97b5db16948b08121ae26"),
}
PROXY_DATASETS = ["marketing_data_full", "Video", "house", "covertype_expanded"]
TEX_PACKAGES = ["libertine.sty", "newtxmath.sty", "pgfplots.sty", "booktabs.sty", "standalone.cls", "pdfpages.sty"]

failures, warnings = [], []


def report(status, what, detail=""):
    print(f"[{status:4}] {what}" + (f"  ({detail})" if detail else ""))
    if status == "FAIL":
        failures.append(what)
    elif status == "WARN":
        warnings.append(what)


def pinned_versions():
    pins = {}
    with open(os.path.join(ARI, "env", "requirements.txt")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==")
                pins[name.strip()] = ver.strip()
    return pins


def check_python_and_packages():
    v = sys.version_info
    report("OK" if v[:2] == (3, 10) else "FAIL", f"Python {platform.python_version()}",
           "" if v[:2] == (3, 10) else "need 3.10: run ARI/install.sh")
    from importlib import metadata
    for name, want in pinned_versions().items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            report("FAIL", f"package {name}", "not installed; run ARI/install.sh")
            continue
        if name == "torch":  # 2.6 or newer: newer GPUs need a newer CUDA build
            ok = [int(x) for x in got.split("+")[0].split(".")[:2]] >= [int(x) for x in want.split(".")[:2]]
        else:
            ok = got == want
        report("OK" if ok else "FAIL", f"package {name} {got}", "" if ok else f"need {want}: run ARI/install.sh")


def check_hardware():
    try:
        import torch
    except ImportError:
        report("FAIL", "GPU", "torch not installed")
        return
    if not torch.cuda.is_available():
        report("WARN", "GPU", "none visible: fine for the default experiments; needed for WITH_BF, WITH_NO_PROXY, "
               "EMBEDDINGS=live and rebuilding the embedding banks")
    else:
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 2**30
        report("OK" if gb >= 24 else "WARN", f"GPU {props.name}, {gb:.0f} GB, CUDA {torch.version.cuda}",
               "" if gb >= 24 else "Llama-3-8B (bf16) + TAPEX + T5 need about 20 GB")
    cpus = os.cpu_count() or 1
    report("OK" if cpus >= 12 else "WARN", f"{cpus} CPU cores", "" if cpus >= 12 else "SAGE+ runs use 12 workers")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        report("OK" if ram >= 64 else "WARN", f"{ram:.0f} GB RAM",
               "" if ram >= 64 else "CoverType (581K x 110) is the largest input")
    except (ValueError, OSError):
        report("WARN", "RAM", "could not read")


def check_files(skip_hash):
    for name, (rows, cols, digest) in DATASETS.items():
        path = os.path.join(ROOT, "data", name)
        if not os.path.exists(path):
            report("FAIL", f"dataset data/{name}", "missing; see 'Data' in README.md")
            continue
        if skip_hash:
            report("OK", f"dataset {name}", "present, hash not checked")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ok = h.hexdigest() == digest
        report("OK" if ok else "FAIL", f"dataset {name} ({rows} rows x {cols} columns)",
               "" if ok else "sha256 differs from the copy used for the paper")
    for ds in PROXY_DATASETS:
        for kind in ("correlation_", "ratio_", "outlier_"):
            p = os.path.join(ROOT, "src", "prompt_learning", "model", f"{ds}_10000_{kind}decision_tree.pkl")
            report("OK" if os.path.exists(p) else "FAIL", f"LLM-proxy {os.path.basename(p)}")
        for suffix in ("attribute_aggfunc_ranks.json", "interesting_attrs.json"):
            p = os.path.join(ROOT, "precompute", f"{ds}_{suffix}")
            report("OK" if os.path.exists(p) else "FAIL", f"precompute/{os.path.basename(p)}")
    banks = [f for f in os.listdir(os.path.join(ROOT, "precompute", "embeddings"))
             if f.endswith(".npz")] if os.path.isdir(os.path.join(ROOT, "precompute", "embeddings")) else []
    report("OK" if len(banks) >= 15 else "FAIL", f"embedding banks precompute/embeddings/ ({len(banks)} of 15)",
           "" if len(banks) >= 15 else "without them every run encodes with T5 / TAPEX (python -m src.diversity.embedding_bank)")
    for name in ("baselines", "daisy.pth"):
        report("OK" if os.path.exists(os.path.join(ROOT, name)) else "FAIL", name)


def check_hf(online):
    token_path = os.environ.get("HF_TOKEN_PATH", os.path.join(ROOT, "hf_token.txt"))
    token = open(token_path).read().strip() if os.path.exists(token_path) else ""
    report("OK" if token else "WARN", f"Hugging Face token file {token_path}",
           "" if token else "missing or empty; put a token with access to meta-llama/Meta-Llama-3-8B-Instruct there")
    cache = os.environ.get("TRANSFORMER_PATH") or os.path.expanduser("~/.cache/huggingface")
    free = shutil.disk_usage(cache if os.path.isdir(cache) else os.path.dirname(cache.rstrip("/")) or "/").free / 2**30
    report("OK" if free >= 30 else "WARN", f"model cache {cache}, {free:.0f} GB free", "" if free >= 30 else "models need about 25 GB")
    try:
        os.environ.setdefault("HF_HOME", cache)
        from huggingface_hub import try_to_load_from_cache, model_info
    except ImportError:
        report("FAIL", "huggingface_hub", "not installed")
        return
    for m in MODELS:
        hit = try_to_load_from_cache(m, "config.json", cache_dir=os.path.join(cache, "hub"))
        report("OK" if isinstance(hit, str) else "WARN", f"model {m} cached", "" if isinstance(hit, str) else ("only for ARI/run_optional.sh: bash ARI/install.sh --llama" if "Llama" in m else "run ARI/scripts/fetch_models.py"))
    if online and token:
        for m in MODELS:
            try:
                model_info(m, token=token)
                report("OK", f"access to {m}")
            except Exception as e:  # gated repo without accepted license, bad token, no network
                report("WARN" if "llama" in m.lower() else "FAIL", f"access to {m}",
                       type(e).__name__ + (" (Llama is only needed for the optional runs)" if "llama" in m.lower() else ""))


def check_latex():
    if shutil.which("pdflatex") is None:
        report("FAIL", "pdflatex", "needed to draw the tables and figures (TeX Live with pgfplots, libertine, newtx)")
        return
    report("OK", "pdflatex")
    if shutil.which("kpsewhich"):
        for pkg in TEX_PACKAGES:
            found = subprocess.run(["kpsewhich", pkg], capture_output=True, text=True).stdout.strip()
            report("OK" if found else "FAIL", f"LaTeX {pkg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true", help="check Hugging Face access to the models")
    ap.add_argument("--render-only", action="store_true", help="only check what drawing the figures needs")
    ap.add_argument("--skip-hash", action="store_true", help="do not hash the datasets (saves ~10 s)")
    args = ap.parse_args()

    print(f"SAGE repository: {ROOT}")
    check_latex()
    if not args.render_only:
        check_python_and_packages()
        check_hardware()
        check_files(args.skip_hash)
        check_hf(args.online)
    print(f"\n{len(failures)} FAIL, {len(warnings)} WARN")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
