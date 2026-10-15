#!/usr/bin/env python3
"""Download T5 and TAPEX (and Llama-3 with --llama) into $TRANSFORMER_PATH (default ~/.cache/huggingface).

    python ARI/scripts/fetch_models.py [--llama]
"""
import os
import sys

ARI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ARI)
LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
MODELS = {
    "microsoft/tapex-large": [],
    "gaussalgo/T5-LM-Large-text2sql-spider": [],
    LLAMA: ["original/*"],  # skip the duplicate consolidated.pth checkpoint
}


def main():
    cache = os.environ.get("TRANSFORMER_PATH")
    if not cache:
        sys.exit("Set TRANSFORMER_PATH to the folder that should hold the models, e.g. TRANSFORMER_PATH=~/hf_cache")
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache  # must be set before huggingface_hub is imported
    from huggingface_hub import snapshot_download

    with_llama = "--llama" in sys.argv[1:]
    token_path = os.environ.get("HF_TOKEN_PATH", os.path.join(ROOT, "hf_token.txt"))
    token = open(token_path).read().strip() if os.path.exists(token_path) else None
    if with_llama and not token:
        sys.exit(f"--llama: {token_path} is missing or empty; put a Hugging Face token with Llama-3 access there")

    for repo, ignore in MODELS.items():
        if repo == LLAMA and not with_llama:
            print(f"skipping {repo} (only for the optional runs; add --llama)")
            continue
        print(f"downloading {repo} -> {cache}")
        path = snapshot_download(repo, token=token, ignore_patterns=ignore or None)
        print(f"  ok: {path}")


if __name__ == "__main__":
    main()
