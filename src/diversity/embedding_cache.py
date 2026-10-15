"""Precompute the T5 query / TAPEX table embeddings of the candidate pivot tables and save them as .npz,
so a later greedy (or brute-force) selection can load them instead of running the encoders.

Run it after pruning is decided, with the same arguments as the run that will use the embeddings:

    python -m src.diversity.embedding_cache --script greedy_basic --filename Video --filepath ./data/ \
        --data_num 0.1 --max_columns 90 --int_threshold 0.5 --do_prune true \
        --transformer_path ~/hf_cache --output_dir precompute/embeddings/

    python -m src.diversity.embedding_cache ... --dry_run true   # only rebuild and list the candidates (no GPU)

It rebuilds the candidate list exactly as the chosen script does before its embedding step:
  read + preprocess the CSV -> canonical combinations -> prune (src.helper.prune_combinations)
  -> greedy.py only: seeded 20% sample -> row selection (first rows; bruteforce.py: df.sample)
  -> SQL text + pivot table (first 100 x 100 cells) per candidate, in the same order.
It then embeds them with src.diversity.helper.get_all_embeddings and batch size 32, as the runs do.
Embeddings are mean-pooled over padded batches, so a vector depends slightly on its batch neighbours:
the saved vectors equal the ones a run computes only for the same candidate list in the same order.
theta and k do not change that list, so one file serves every theta/k of a configuration.

Output: <output_dir>/<filename>_<script>_dn<data_num>_mc<max_columns>_it<int_threshold>_pr<do_prune>.npz
  keys   SQL text of each candidate, in run order
  query  T5 query embeddings      (n x 1024, float32)
  table  TAPEX table embeddings   (n x 1024, float32)
  meta   JSON: the arguments above, models, batch size, n, embedding_time_sec (encoding time, so it can
         be added back to runtimes measured with cached embeddings)

Use in a run:  combined = load_combined(path, sql_queries)  # same matrix as concat_embeddings(...)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

QUERY_CHECKPOINT = "gaussalgo/T5-LM-Large-text2sql-spider"
TABLE_CHECKPOINT = "microsoft/tapex-large"
BATCH_SIZE = 32  # the batch size the runs pass to compute_and_concat_embeddings_with_timing
SCRIPTS = ("greedy_basic", "greedy", "bruteforce")


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", choices=SCRIPTS, default="greedy_basic",
                    help="run whose candidate list to rebuild: greedy_basic (SAGE), greedy (SAGE+), bruteforce")
    ap.add_argument("--filename", required=True, help="CSV name without .csv")
    ap.add_argument("--filepath", default="./data/", help="folder of the CSV (with trailing /)")
    ap.add_argument("--transformer_path", default=os.environ.get("TRANSFORMER_PATH", ""), help="model cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--int_threshold", type=float, default=0.5)
    ap.add_argument("--data_num", type=float, default=0.2)
    ap.add_argument("--max_columns", type=int, default=None)
    ap.add_argument("--do_prune", type=str2bool, default=True)
    ap.add_argument("--do_int_cache", type=str2bool, default=True)
    ap.add_argument("--interesting_attributes_path", default=None,
                    help="default precompute/<filename>_interesting_attrs.json")
    ap.add_argument("--attribute_aggfunc_ranks", default=None,
                    help="default precompute/<filename>_attribute_aggfunc_ranks.json")
    ap.add_argument("--output_dir", default="precompute/embeddings/")
    ap.add_argument("--dry_run", type=str2bool, default=False, help="rebuild and list the candidates only")
    return ap.parse_args(argv)


def cache_name(args):
    return (f"{args.filename}_{args.script}_dn{args.data_num:g}_mc{args.max_columns}"
            f"_it{args.int_threshold:g}_pr{args.do_prune}.npz")


def load_dataframe(filepath, filename, max_columns=None):
    """The CSV read and preprocessed as the scripts do. Preprocess works column by column, so
    load_dataframe(...)[first n columns] equals load_dataframe(..., max_columns=n)."""
    from src.preprocessing import Preprocess
    df = pd.read_csv(f"{filepath}{filename}.csv", index_col=False).dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].fillna(0)
    if max_columns is not None and max_columns < df.shape[1]:
        df = df.iloc[:, :max_columns]
    return Preprocess(df).do_preprocess()


def candidate_inputs(args, prune_dir, df_full=None):
    """(combs, sql_queries, pivot_tables) in the order the chosen script embeds them.
    df_full: the whole dataset from load_dataframe(), to avoid re-reading it for every configuration."""
    from src.generate_table import generate_canonical_combinations
    from src.helper import prune_combinations, set_random_seed
    from src.utility.utility import get_precomputing_saved_result

    set_random_seed(args.seed)  # same seeding as the scripts; greedy.py's sample depends on it
    os.makedirs(prune_dir, exist_ok=True)  # prune_combinations writes pruned.csv there

    if df_full is None:
        df = load_dataframe(args.filepath, args.filename, args.max_columns)
    elif args.max_columns is not None and args.max_columns < df_full.shape[1]:
        df = df_full.iloc[:, :args.max_columns].copy()
    else:
        df = df_full.copy()

    result = generate_canonical_combinations(df, df.columns)
    combs = result[0] + result[1]

    attrs_path = args.interesting_attributes_path or f"precompute/{args.filename}_interesting_attrs.json"
    ranks_path = args.attribute_aggfunc_ranks or f"precompute/{args.filename}_attribute_aggfunc_ranks.json"
    if not (os.path.exists(attrs_path) and os.path.exists(ranks_path)):
        sys.exit(f"{attrs_path} / {ranks_path} not found: run the experiment once with Llama to create them")
    interesting_attrs, unique_values_dict, aggfunc_ranks = get_precomputing_saved_result(
        df=df, interesting_attrs_path=attrs_path, attribute_aggfunc_ranks_path=ranks_path)

    if args.do_prune:
        kwargs = {} if args.script == "bruteforce" else {"do_int_cache": args.do_int_cache}
        pruned = prune_combinations(
            combs=combs, interesting_attrs=interesting_attrs, df=df,
            unique_values_dict=unique_values_dict, aggfunc_ranks=aggfunc_ranks,
            int_threshold=args.int_threshold, output_dir=prune_dir, **kwargs)
        if not isinstance(pruned, list):  # dict with caches when the interpretability cache is on
            pruned = pruned.get("pruned", [])
        if args.script == "bruteforce" and len(combs) >= 10000:
            print("[warn] bruteforce.py prunes in parallel for >= 10,000 combinations when --do_parallel is "
                  "true; its candidate order may then differ from this list")
    else:
        pruned = combs

    if args.script == "greedy":  # SAGE+: same sample as src/models/greedy.py
        frac = 1.0 if "video" in args.filename.lower() else 0.2
        n_pts = max(50, int(len(pruned) * frac))
        pruned = random.sample(pruned, min(len(pruned), n_pts))

    if args.script == "bruteforce":  # bruteforce.py samples rows; the greedy scripts take the first rows
        if args.data_num <= 1:
            df = df.sample(frac=args.data_num, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=int(args.data_num), random_state=42).reset_index(drop=True)
    elif args.data_num <= 1:
        df = df.iloc[:int(len(df) * args.data_num)].reset_index(drop=True)
    else:
        df = df.iloc[:int(args.data_num)].reset_index(drop=True)

    sql_queries, pivot_tables = [], []
    for comb in pruned:
        sql, pt = query_and_pivot(comb, df)
        sql_queries.append(sql)
        pivot_tables.append(pt)
    return pruned, sql_queries, pivot_tables


def query_and_pivot(comb, df):
    """SQL text and pivot table of one candidate, built as in src.utility.helper.get_single_utilty."""
    index = comb["index"]
    columns = comb["column"]
    value = comb["value"]
    aggfunc = comb["aggfunc"].upper()
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        columns = []
    group_by = [index] + columns if columns else [index]
    value_clause = [f"{aggfunc}({v})" for v in value] if isinstance(value, list) else [f"{aggfunc}({value})"]
    sql_query = f"SELECT {', '.join(group_by + value_clause)} FROM table GROUP BY {', '.join(group_by)}"
    pivot = pd.pivot_table(df, index=index, columns=columns, values=value, aggfunc=aggfunc.lower())
    return sql_query, pivot.iloc[:100, :100]


def compute_embeddings(sql_queries, pivot_tables, transformer_path):
    import torch
    from src.diversity.helper import get_all_embeddings
    from src.helper import load_encoders

    if transformer_path:
        for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
            os.environ[var] = transformer_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_tok, q_model, t_tok, t_model = load_encoders(QUERY_CHECKPOINT, TABLE_CHECKPOINT, device)
    start = time.time()
    with torch.no_grad():
        query, table = get_all_embeddings(sql_queries, pivot_tables, q_tok, q_model, t_tok, t_model,
                                          device, BATCH_SIZE)
    return np.vstack(query).astype(np.float32), np.vstack(table).astype(np.float32), time.time() - start, device


def save(path, sql_queries, query, table, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, keys=np.array(sql_queries), query=query, table=table, meta=json.dumps(meta))


def load(path, sql_queries=None):
    """(query, table, meta). With sql_queries, rows are returned in that order; every query must be in the file.
    A different order or subset is allowed but warned about: those vectors were computed in other batches."""
    with np.load(path, allow_pickle=False) as z:
        keys, query, table, meta = list(z["keys"]), z["query"], z["table"], json.loads(str(z["meta"]))
    if sql_queries is None or list(sql_queries) == keys:
        return query, table, meta
    pos = {k: i for i, k in enumerate(keys)}
    missing = [q for q in sql_queries if q not in pos]
    if missing:
        raise KeyError(f"{len(missing)} queries are not in {path}, e.g. {missing[0]}")
    print(f"[embedding_cache] {path}: candidate order differs from the cached run; vectors were computed "
          "in other batches and may differ slightly from a fresh computation")
    idx = [pos[q] for q in sql_queries]
    return query[idx], table[idx], meta


def load_combined(path, sql_queries):
    """The matrix concat_embeddings() would return for these queries (query | table / 2)."""
    from src.diversity.helper import concat_embeddings
    query, table, _ = load(path, sql_queries)
    return concat_embeddings(list(query), list(table))


def main(argv=None):
    args = parse_args(argv)
    path = os.path.join(args.output_dir, cache_name(args))
    prune_dir = os.path.join(args.output_dir, "prune", os.path.splitext(cache_name(args))[0])
    os.makedirs(prune_dir, exist_ok=True)

    combs, sql_queries, pivot_tables = candidate_inputs(args, prune_dir)
    print(f"[embedding_cache] {len(sql_queries)} candidates for {args.script} on {args.filename}")
    if args.dry_run:
        for i, comb in enumerate(combs):
            print(f"  {i}\t{comb}")
        return

    query, table, secs, device = compute_embeddings(sql_queries, pivot_tables, args.transformer_path)
    meta = {k: v for k, v in vars(args).items() if k not in ("transformer_path", "output_dir", "dry_run")}
    meta.update(n=len(sql_queries), batch_size=BATCH_SIZE, device=device, embedding_time_sec=round(secs, 4),
                query_model=QUERY_CHECKPOINT, table_model=TABLE_CHECKPOINT,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(path, sql_queries, query, table, meta)
    print(f"[embedding_cache] {len(sql_queries)} x ({query.shape[1]} + {table.shape[1]}) in {secs:.1f} s -> {path}")


if __name__ == "__main__":
    main()
