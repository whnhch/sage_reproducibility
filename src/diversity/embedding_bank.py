"""Embedding bank: T5 (query) and TAPEX (table) vectors of every candidate pivot table, encoded once and
indexed by greedy_basic.py / greedy.py / bruteforce.py runs instead of running the encoders.

    python -m src.diversity.embedding_bank --dry_run true                 # what would be built (no models)
    python -m src.diversity.embedding_bank --transformer_path ~/hf_cache  # build or extend all banks (GPU)
    python -m src.diversity.embedding_bank --datasets Video house         # only these datasets
    python -m src.diversity.embedding_bank --verify true                  # also check against direct encoding

Runs mean-pool over padded batches of 32, so a vector depends on its batch's padded length L (TAPEX attends
to padding). A bank stores each candidate's vectors at every L the runs in PAPER_PLAN produce, so indexing
returns exactly the vectors a run computes. One bank per dataset and row selection:
<output_dir>/<dataset>_<head|sample>_dn<data_num>.npz (head: first rows; sample: bruteforce.py's df.sample).
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from src.diversity import embedding_cache as ec

BATCH_SIZE = ec.BATCH_SIZE
QUERY_TEXT = "Summarize the table"  # TAPEX query used by get_all_embeddings

# Per-dataset settings of paper_figures/experiments_env.sh: (data_num, max_columns, int_threshold)
DEFAULTS = {"marketing_data_full": (1.0, 90, 0.5), "Video": (0.1, 90, 0.5), "house": (1.0, 90, 0.5),
            "covertype_expanded": (1.0, 200, 0.6)}
# Pruning threshold of the 5-attribute ablation (paper_figures/table9 and fig10/run_experiments.sh)
ABLATION_INT_THRESHOLD = {"marketing_data_full": 0.5, "Video": 0.3, "house": 0.3}


def paper_plan(params=None):
    """Every (script, dataset, data_num, max_columns, int_threshold, do_prune) the experiments run.
    params="paper_text" (default: $PARAMS): pruning threshold 0.5 everywhere (PARAMS in experiments_env.sh)."""
    plan = []
    defaults, ablation = dict(DEFAULTS), dict(ABLATION_INT_THRESHOLD)
    if (params or os.environ.get("PARAMS", "code")) == "paper_text":
        defaults["covertype_expanded"] = (1.0, 200, 0.5)
        ablation = {ds: 0.5 for ds in ablation}

    def add(script, ds, dn, mc, it, prune):
        plan.append(dict(script=script, filename=ds, data_num=dn, max_columns=mc, int_threshold=it, do_prune=prune))

    for ds in ("marketing_data_full", "Video", "house"):
        dn, mc, it = defaults[ds]
        add("greedy_basic", ds, dn, mc, it, True)   # Table 8 SAGE + Top-k, Fig. 10 right (Video)
        add("greedy", ds, dn, mc, it, True)         # Table 8 SAGE+
        add("bruteforce", ds, dn, 5, 0.0, False)    # Table 9, Fig. 10 left
        add("greedy_basic", ds, dn, 5, ablation[ds], False)  # -PR&PX, -PR
        add("greedy_basic", ds, dn, 5, ablation[ds], True)   # -PX, SAGE
    add("greedy_basic", "Video", 1.0, defaults["Video"][1], defaults["Video"][2], True)  # Fig. 10 right: all rows
    for ds, n_cols in (("marketing_data_full", 28), ("covertype_expanded", 110)):  # Fig. 11
        _, mc, it = defaults[ds]
        for script in ("greedy_basic", "greedy"):
            for pct in (20, 40, 60, 80, 100):
                add(script, ds, pct / 100, mc, it, True)                      # tuples
                add(script, ds, 1.0, round(pct / 100 * n_cols), it, True)     # attributes
    unique = {json.dumps(c, sort_keys=True): c for c in plan}
    return list(unique.values())


def rows_of(script, filename=None, data_num=1.0):
    """Row selection of a run: bruteforce.py samples; the greedy scripts take the first rows, except on Video."""
    sample = script == "bruteforce" or (filename == "Video" and data_num <= 1)
    return "sample" if sample else "head"


def bank_name(filename, rows, data_num):
    return f"{filename}_{rows}_dn{data_num:g}.npz"


def bank_path(args, script, output_dir="precompute/embeddings/"):
    """Bank file a run with these args (filename, data_num) uses."""
    return os.path.join(output_dir, bank_name(args.filename, rows_of(script, args.filename, args.data_num), args.data_num))


def config_args(cfg, filepath):
    argv = ["--script", cfg["script"], "--filename", cfg["filename"], "--filepath", filepath,
            "--data_num", str(cfg["data_num"]), "--max_columns", str(cfg["max_columns"]),
            "--int_threshold", str(cfg["int_threshold"]), "--do_prune", str(cfg["do_prune"])]
    return ec.parse_args(argv)


# --------------------------------------------------------------------------- #
# Bank storage
# --------------------------------------------------------------------------- #
class Bank:
    def __init__(self):
        self.keys, self.q_len, self.t_len = [], [], []
        self.q, self.t = {}, {}  # (key index, L) -> vector
        self.meta = {"configs": [], "embedding_time_sec": 0.0}
        self.pos = {}

    @classmethod
    def load(cls, path):
        b = cls()
        if not os.path.exists(path):
            return b
        with np.load(path, allow_pickle=False) as z:
            b.keys = [str(k) for k in z["keys"]]
            b.q_len, b.t_len = [int(x) for x in z["q_len"]], [int(x) for x in z["t_len"]]
            b.q = {(int(i), int(L)): v for i, L, v in zip(z["q_key"], z["q_L"], z["q_vec"])}
            b.t = {(int(i), int(L)): v for i, L, v in zip(z["t_key"], z["t_L"], z["t_vec"])}
            b.meta = json.loads(str(z["meta"]))
        b.pos = {k: i for i, k in enumerate(b.keys)}
        return b

    def add_key(self, key, q_len, t_len):
        if key not in self.pos:
            self.pos[key] = len(self.keys)
            self.keys.append(key)
            self.q_len.append(q_len)
            self.t_len.append(t_len)
        return self.pos[key]

    def save(self, path):
        def stack(d):
            items = sorted(d.items())
            idx = np.array([i for (i, _), _ in items], dtype=np.int32)
            lens = np.array([L for (_, L), _ in items], dtype=np.int32)
            vec = np.stack([v for _, v in items]).astype(np.float32) if items else np.zeros((0, 1024), np.float32)
            return idx, lens, vec
        q_key, q_L, q_vec = stack(self.q)
        t_key, t_L, t_vec = stack(self.t)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npz"  # per process, so concurrent writers cannot mix their files
        np.savez_compressed(tmp, keys=np.array(self.keys), q_len=np.array(self.q_len, np.int32),
                            t_len=np.array(self.t_len, np.int32), q_key=q_key, q_L=q_L, q_vec=q_vec,
                            t_key=t_key, t_L=t_L, t_vec=t_vec, meta=json.dumps(self.meta))
        os.replace(tmp, path)  # never leave a half-written bank behind


def batch_lengths(idx, lengths, batch_size=BATCH_SIZE):
    """For candidates in run order, the padded length L each one is embedded at."""
    out = []
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        L = max(lengths[i] for i in chunk)
        out += [L] * len(chunk)
    return out


def run_vectors(path, sql_queries, batch_size=BATCH_SIZE, bank=None):
    """T5 and TAPEX vectors (lists of arrays) exactly as get_all_embeddings would return them for this run."""
    b = bank or Bank.load(path)
    missing = [q for q in sql_queries if q not in b.pos]
    if missing:
        raise KeyError(f"{len(missing)} candidates are not in {path}, e.g. {missing[0]}; add the run's "
                       "configuration to the plan and rebuild")
    idx = [b.pos[q] for q in sql_queries]
    q_out, t_out = [], []
    for i, Lq, Lt in zip(idx, batch_lengths(idx, b.q_len, batch_size), batch_lengths(idx, b.t_len, batch_size)):
        if (i, Lq) not in b.q or (i, Lt) not in b.t:
            raise KeyError(f"{path} has no vector for '{b.keys[i]}' at padded length T5 {Lq} / TAPEX {Lt}: this run "
                           "order was not planned; add its configuration and rebuild")
        q_out.append(b.q[(i, Lq)])
        t_out.append(b.t[(i, Lt)])
    return q_out, t_out


def run_combined(path, sql_queries, batch_size=BATCH_SIZE):
    """The matrix concat_embeddings() returns in the run (query | table / 2)."""
    from src.diversity.helper import concat_embeddings
    q, t = run_vectors(path, sql_queries, batch_size)
    return concat_embeddings(q, t)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
class Encoders:
    def __init__(self, transformer_path, load_models):
        import torch
        if transformer_path:
            for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
                os.environ[var] = transformer_path
        from transformers import AutoTokenizer, TapexTokenizer
        self.torch = torch
        self.q_tok = AutoTokenizer.from_pretrained(ec.QUERY_CHECKPOINT)
        self.t_tok = TapexTokenizer.from_pretrained(ec.TABLE_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_model = self.t_model = None
        if load_models:
            from src.helper import load_encoders
            _, self.q_model, _, self.t_model = load_encoders(ec.QUERY_CHECKPOINT, ec.TABLE_CHECKPOINT, self.device)

    def table_inputs(self, pivot):
        """Tokenized table exactly as get_all_embeddings does it (unpadded)."""
        pt = pivot.copy()
        pt.columns = [str(c) if c is not None else "" for c in pt.columns]
        return self.t_tok(table=pt.astype(str), query=QUERY_TEXT, return_tensors="pt", padding=True, truncation=True)

    def q_length(self, sql):
        return len(self.q_tok(sql, truncation=True)["input_ids"])

    def query_vectors(self, sqls, L):
        """T5 vectors of queries padded to L (padding masked, as with padding=True in a batch whose max is L)."""
        inp = self.q_tok(sqls, return_tensors="pt", padding="max_length", max_length=L, truncation=True).to(self.device)
        with self.torch.no_grad():
            return self.q_model(**inp).last_hidden_state.mean(dim=1).cpu().numpy()

    def table_vectors(self, encs, L):
        """TAPEX vectors of tables padded to L, with input_ids and attention_mask padded by the pad id like
        get_all_embeddings' pad_sequence collate."""
        F, pad = self.torch.nn.functional, self.t_tok.pad_token_id
        batch = {key: self.torch.stack([F.pad(e[key][0], (0, L - e[key].shape[1]), value=pad) for e in encs]).to(self.device)
                 for key in encs[0]}
        with self.torch.no_grad():
            return self.t_model.model.encoder(**batch).last_hidden_state.mean(dim=1).cpu().numpy()


def plan_bank(configs, df_full, enc, bank, prune_root):
    """Add every candidate of these configs to the bank; return the (candidate, L) pairs still missing
    and, per config, the run-order SQL list (for --verify)."""
    pivots, need_q, need_t, orders = {}, set(), set(), []
    for cfg in configs:
        args = config_args(cfg, "./data/")
        tag = f"{cfg['script']}_mc{cfg['max_columns']}_it{cfg['int_threshold']:g}_pr{cfg['do_prune']}"
        _, sqls, pts = ec.candidate_inputs(args, os.path.join(prune_root, tag), df_full)
        idx = []
        for sql, pt in zip(sqls, pts):
            if sql not in bank.pos:
                bank.add_key(sql, enc.q_length(sql), int(enc.table_inputs(pt)["input_ids"].shape[1]))
            i = bank.pos[sql]
            pivots.setdefault(i, pt)
            idx.append(i)
        for i, L in zip(idx, batch_lengths(idx, bank.q_len)):
            if (i, L) not in bank.q:
                need_q.add((i, L))
        for i, L in zip(idx, batch_lengths(idx, bank.t_len)):
            if (i, L) not in bank.t:
                need_t.add((i, L))
        orders.append((cfg, sqls, pts))
        print(f"  {tag}: {len(sqls)} candidates")
        if cfg not in bank.meta["configs"]:
            bank.meta["configs"].append(cfg)
    return pivots, need_q, need_t, orders


def fill_bank(bank, pivots, need_q, need_t, enc, path, checkpoint_sec=600):
    start, last_save = time.time(), time.time()
    by_L = defaultdict(list)
    for i, L in sorted(need_q):
        by_L[L].append(i)
    for L, ids in sorted(by_L.items()):
        for s in range(0, len(ids), BATCH_SIZE):
            chunk = ids[s:s + BATCH_SIZE]
            for i, v in zip(chunk, enc.query_vectors([bank.keys[i] for i in chunk], L)):
                bank.q[(i, L)] = v
    print(f"  T5: {len(need_q)} vectors in {time.time() - start:.0f} s")

    t0 = time.time()
    by_L = defaultdict(list)
    for i, L in sorted(need_t):
        by_L[L].append(i)
    done = 0
    for L, ids in sorted(by_L.items()):
        for s in range(0, len(ids), BATCH_SIZE):
            chunk = ids[s:s + BATCH_SIZE]
            for i, v in zip(chunk, enc.table_vectors([enc.table_inputs(pivots[i]) for i in chunk], L)):
                bank.t[(i, L)] = v
            done += len(chunk)
        if time.time() - last_save > checkpoint_sec:  # save progress periodically
            bank.meta["embedding_time_sec"] += time.time() - start
            start = last_save = time.time()
            bank.save(path)
            print(f"  checkpoint: {done}/{len(need_t)} TAPEX vectors saved")
    print(f"  TAPEX: {len(need_t)} vectors in {time.time() - t0:.0f} s")
    bank.meta["embedding_time_sec"] += time.time() - start
    bank.meta.update(device=enc.device, batch_size=BATCH_SIZE, query_model=ec.QUERY_CHECKPOINT,
                     table_model=ec.TABLE_CHECKPOINT, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    bank.save(path)


def verify(path, orders, enc):
    """Compare indexed vectors with get_all_embeddings run directly in each configuration's order."""
    from src.diversity.helper import get_all_embeddings
    bank = Bank.load(path)
    for cfg, sqls, pts in orders:
        with enc.torch.no_grad():
            q_ref, t_ref = get_all_embeddings(sqls, [p.copy() for p in pts], enc.q_tok, enc.q_model,
                                              enc.t_tok, enc.t_model, enc.device, BATCH_SIZE)
        q_bank, t_bank = run_vectors(path, sqls, bank=bank)
        rel = lambda a, b: max(float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12)) for x, y in zip(a, b))
        print(f"  verify {cfg['script']} mc{cfg['max_columns']} pr{cfg['do_prune']}: max relative diff "
              f"T5 {rel(q_bank, q_ref):.1e}, TAPEX {rel(t_bank, t_ref):.1e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all datasets in the plan")
    ap.add_argument("--transformer_path", default=os.environ.get("TRANSFORMER_PATH", ""))
    ap.add_argument("--output_dir", default="precompute/embeddings/")
    ap.add_argument("--dry_run", type=ec.str2bool, default=False, help="list banks and missing vectors only")
    ap.add_argument("--verify", type=ec.str2bool, default=False, help="after building, compare with direct runs")
    args = ap.parse_args(argv)

    plan = paper_plan()
    datasets = args.datasets or list(dict.fromkeys(c["filename"] for c in plan))
    enc = Encoders(args.transformer_path, load_models=not args.dry_run)
    print(f"device: {enc.device}")
    for ds in datasets:
        configs = [c for c in plan if c["filename"] == ds]
        if not configs:
            print(f"{ds}: not in the plan")
            continue
        print(f"== {ds}: reading and preprocessing")
        df_full = ec.load_dataframe("./data/", ds)
        groups = defaultdict(list)
        for c in configs:
            groups[(rows_of(c["script"], ds, c["data_num"]), c["data_num"])].append(c)
        for (rows, dn), cfgs in sorted(groups.items()):
            path = os.path.join(args.output_dir, bank_name(ds, rows, dn))
            print(f"-- {path} ({len(cfgs)} configurations)")
            bank = Bank.load(path)
            pivots, need_q, need_t, orders = plan_bank(cfgs, df_full, enc, bank,
                                                       os.path.join(args.output_dir, "prune", f"{ds}_{rows}_dn{dn:g}"))
            print(f"  {len(bank.keys)} candidates; missing {len(need_q)} T5 and {len(need_t)} TAPEX vectors "
                  f"(TAPEX lengths {min(bank.t_len)}-{max(bank.t_len)} tokens)")
            if args.dry_run:
                continue
            if need_q or need_t:
                fill_bank(bank, pivots, need_q, need_t, enc, path)
                print(f"  saved {path} ({os.path.getsize(path) / 2**20:.1f} MB)")
            if args.verify:
                verify(path, orders, enc)


if __name__ == "__main__":
    main()
