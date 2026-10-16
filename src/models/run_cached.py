"""Run greedy_basic.py, greedy.py, bruteforce.py or software.py without loading models they do not use.

    python -m src.models.run_cached greedy_basic [--embeddings bank|live] [--bank_dir DIR] [--clf_rat_path PKL]
        <the script's own arguments>

The script is imported unchanged; these names in it are replaced:
  Llama3            loaded on first use, so runs with the LLM-proxy never load it.
  load_encoders,    --embeddings bank (default): T5 / TAPEX vectors come from precompute/embeddings/
  compute_and_      (src/diversity/embedding_bank.py), so runtimes exclude encoding; the encoders load only if a
  concat_embeddings vector is missing. --embeddings live, and software.py, always encode.
  _with_timing
  set_environment   reads the Hugging Face token only if present, without printing it.
  load_classifiers  with --clf_rat_path: --clf_trd_path is the correlation tree, combined with the ratio tree
                    (src/prompt_learning/proxy.py).
score_as_number returns NaiveInsightfulness's [0] scores as 0.0, so max() does not fail on 1 x 1 tables.
"""
import csv
import importlib
import os
import sys
import time

SCRIPTS = ("greedy_basic", "greedy", "bruteforce", "software")


def score_as_number(method):
    """Wrap a NaiveInsightfulness score method so a [0] result becomes 0.0."""
    def wrapped(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        return float(value[0]) if isinstance(value, list) and len(value) == 1 else value
    wrapped.__wrapped__ = method
    return wrapped


def patch_scoring():
    from src.utility import insightfulness
    cls = insightfulness.NaiveInsightfulness
    for name in ("calculate_inf", "calculate_correlation", "calculate_ratio", "calculate_surprising"):
        method = getattr(cls, name)
        if not hasattr(method, "__wrapped__"):
            setattr(cls, name, score_as_number(method))


# At import time, so SAGE+'s spawned workers get it too.
patch_scoring()


class LazyLlama:
    """Stands in for Llama3(): creates the real model the first time any of its attributes is used."""
    real_class = None

    def __init__(self, *args, **kwargs):
        self._init = (args, kwargs)
        self._llm = None

    def _load(self):
        if self._llm is None:
            print("[run_cached] loading Llama-3 (first use)", flush=True)
            self._llm = LazyLlama.real_class(*self._init[0], **self._init[1])
        return self._llm

    def __getattr__(self, name):
        return getattr(self._load(), name)


def set_environment(transformer_path, hf_token_path):
    """src.helper.set_environment, without requiring or printing the token."""
    for var in ("TRANSFORMERS_CACHE", "HF_HOME", "HF_DATASETS_CACHE"):
        os.environ[var] = transformer_path
    token = open(hf_token_path).read().strip() if hf_token_path and os.path.exists(hf_token_path) else ""
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN has been set from", hf_token_path)
    else:
        print("no Hugging Face token file; only needed if Llama-3 has to be downloaded")


def bank_embeddings(script, bank_dir, real_load_encoders, real_compute):
    """Replacement for compute_and_concat_embeddings_with_timing that indexes the embedding bank."""
    from src.diversity import embedding_bank as eb

    def compute(sql_queries, pivot_tables, query_tokenizer, query_model, table_tokenizer, table_model,
                device, batch_size, output_dir, concat_embeddings_fn):
        os.makedirs(output_dir, exist_ok=True)
        args = compute.args
        path = eb.bank_path(args, script, bank_dir)
        start = time.time()
        try:
            q, t = eb.run_vectors(path, sql_queries, batch_size=batch_size)
            source = path
        except KeyError as e:
            print(f"[run_cached] {e}\n[run_cached] computing the embeddings with T5 / TAPEX instead", flush=True)
            qt, qm, tt, tm = real_load_encoders(QUERY_CHECKPOINT, TABLE_CHECKPOINT, device)
            return real_compute(sql_queries, pivot_tables, qt, qm, tt, tm, device, batch_size, output_dir,
                                concat_embeddings_fn)
        lookup = time.time() - start
        t0 = time.time()
        combined = concat_embeddings_fn(q, t)
        with open(os.path.join(output_dir, "embedding_time.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["embedding_time_sec", "concat_time_sec"])
            w.writeheader()
            w.writerow({"embedding_time_sec": round(lookup, 4), "concat_time_sec": round(time.time() - t0, 4)})
        with open(os.path.join(output_dir, "embedding_source.txt"), "w") as f:
            f.write(f"bank {source}\n(embedding_time_sec is the time to read the vectors, not to encode them)\n")
        print(f"[run_cached] {len(sql_queries)} embeddings read from {source} in {lookup:.2f} s", flush=True)
        return combined

    return compute


QUERY_CHECKPOINT = "gaussalgo/T5-LM-Large-text2sql-spider"
TABLE_CHECKPOINT = "microsoft/tapex-large"


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in SCRIPTS:
        sys.exit(f"usage: python -m src.models.run_cached {{{','.join(SCRIPTS)}}} [--embeddings bank|live] "
                 "[--bank_dir DIR] [--clf_rat_path PKL] <script arguments>")
    script, rest = argv[0], argv[1:]
    mode, bank_dir, rat_path = "bank", "precompute/embeddings/", None
    for opt in ("--embeddings", "--bank_dir", "--clf_rat_path"):
        if opt in rest:
            i = rest.index(opt)
            value = rest[i + 1]
            del rest[i:i + 2]
            if opt == "--embeddings":
                mode = value
            elif opt == "--bank_dir":
                bank_dir = value
            else:
                rat_path = value
    if mode not in ("bank", "live"):
        sys.exit("--embeddings must be bank or live")

    module = importlib.import_module(f"src.models.{script}")
    LazyLlama.real_class = module.Llama3
    module.Llama3 = LazyLlama
    module.set_environment = set_environment
    if rat_path:
        import joblib
        from src.prompt_learning.proxy import TrendProxy
        real_load_classifiers = module.load_classifiers

        def load_classifiers(clf_trd_path, clf_out_path):
            clf_cor, clf_out = real_load_classifiers(clf_trd_path, clf_out_path)
            print(f"[run_cached] LLM-proxies: correlation {clf_trd_path}, ratio {rat_path}, outlier {clf_out_path}",
                  flush=True)
            return TrendProxy(clf_cor, joblib.load(rat_path)), clf_out
        module.load_classifiers = load_classifiers
    elif hasattr(module, "load_classifiers"):
        real_load_classifiers = module.load_classifiers

        def load_classifiers(clf_trd_path, clf_out_path):
            print(f"[run_cached] LLM-proxies: trend {clf_trd_path}, outlier {clf_out_path}", flush=True)
            return real_load_classifiers(clf_trd_path, clf_out_path)
        module.load_classifiers = load_classifiers
    if mode == "bank" and script != "software":
        real_load_encoders = module.load_encoders
        module.load_encoders = lambda *a, **k: (None, None, None, None)  # loaded only if the bank misses
        compute = bank_embeddings(script, bank_dir, real_load_encoders, module.compute_and_concat_embeddings_with_timing)
        module.compute_and_concat_embeddings_with_timing = compute

    sys.argv = [f"src/models/{script}.py"] + rest
    args = module.parse_args()
    if mode == "bank" and script != "software":
        module.compute_and_concat_embeddings_with_timing.args = args
    print(f"[run_cached] {script}: Llama-3 loaded on first use; embeddings: "
          f"{'bank ' + bank_dir if mode == 'bank' and script != 'software' else 'computed'}", flush=True)
    module.main(args)


if __name__ == "__main__":
    main()
