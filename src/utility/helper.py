import os
import pandas as pd
from src.utility.llama import Prompt
from src.utility.utility import Utility
import csv
import time
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
import math

def worker_wrapper(args):
    (
        sub_combs, df, interesting_attrs, aggfunc_ranks,
        clf_trd, clf_out, args_obj, output_dir,
        unique_dict, gamma, tau_cor, tau_rat, prune,
        concise_cache, semantics_cache
    ) = args

    # Call your existing function, suppress saving in subprocess
    return compute_utility_with_embeddings(
        combs=sub_combs,
        df=df,
        llm=None,  # Don't pass LLM to multiprocessing
        interesting_attrs=interesting_attrs,
        aggfunc_ranks=aggfunc_ranks,
        clf_trd=clf_trd,
        clf_out=clf_out,
        args=args_obj,
        output_dir=output_dir,  # Same output_dir is OK if not writing per process
        save_sql=False,
        save_pivot=False,
        verbose=False,
        unique_dict=unique_dict,
        gamma=gamma,
        tau_cor=tau_cor,
        tau_rat=tau_rat,
        prune=prune
    )
def parallel_compute_utility(
    combs, df, llm, interesting_attrs, aggfunc_ranks,
    clf_trd, clf_out, args_obj, output_dir, unique_dict,
    gamma, tau_cor, tau_rat, prune,
    num_workers=4, do_parallel = False,
    concise_cache:dict=None, semantics_cache:dict=None,
):
    start_time = time.time()  # ⏱ Start timing

    if llm is not None or not do_parallel:
        print("[Info] LLM detected — running sequentially.")
        return compute_utility_with_embeddings(
            combs=combs,
            df=df,
            llm=llm,
            interesting_attrs=interesting_attrs,
            aggfunc_ranks=aggfunc_ranks,
            clf_trd=clf_trd,
            clf_out=clf_out,
            args=args_obj,
            output_dir=output_dir,
            save_sql=True,
            save_pivot=True,
            verbose=True,
            unique_dict=unique_dict,
            gamma=gamma,
            tau_cor=tau_cor,
            tau_rat=tau_rat,
            prune=prune,
            do_parallel=do_parallel,
            concise_cache=concise_cache, semantics_cache=semantics_cache,
        )

    print(f"[Info] Running in parallel with {num_workers} workers.")
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = math.ceil(len(combs) / num_workers)
    chunks = [combs[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    pool_args = [
        (
            chunk, df, interesting_attrs, aggfunc_ranks,
            clf_trd, clf_out, args_obj, output_dir,
            unique_dict, gamma, tau_cor, tau_rat, prune,
            concise_cache, semantics_cache
        )
        for chunk in chunks
    ]

    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        results = pool.map(worker_wrapper, pool_args)

    all_results, all_sql_queries, all_pivots = [], [], []
    for r, s, p in results:
        all_results.extend(r)
        all_sql_queries.extend(s)
        all_pivots.extend(p)

    if all_results:
        keys = all_results[0].keys()
        with open(os.path.join(output_dir, "utility_scores.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    elapsed = time.time() - start_time 
    with open(os.path.join(output_dir, "utility_total_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Computed combinations", len(all_results)])
        writer.writerow(["Total elapsed time (seconds)", f"{elapsed:.4f}"])

    return all_results, all_sql_queries, all_pivots

def get_single_utilty(comb, df, llm, 
                      interesting_attrs, aggfunc_ranks,
                      clf_trd, clf_out,
                      unique_dict,
                      output_dir,
                      gamma:float, tau_cor:float, tau_rat:float,
                      prune:bool=True,
                      concise_cache:dict=None, semantics_cache:dict=None,
                      ):
    index = comb["index"]
    columns = comb["column"]
    value = comb["value"]
    aggfunc = comb["aggfunc"].upper()
    
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        columns = []
            
    group_by = [index] + columns if columns else [index]
    
    if isinstance(value, list):
        value_clause = [f"{aggfunc}({v})" for v in value]
    else:
        value_clause = [f"{aggfunc}({value})"]

    select_clause = ", ".join(group_by + value_clause)

#     select_clause = ", ".join(group_by + [f"{aggfunc}({value})"])
    group_by_clause = ", ".join(group_by)
    sql_query = f"SELECT {select_clause} FROM table GROUP BY {group_by_clause}"
    
    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=value,
        aggfunc=aggfunc.lower()  # pandas aggfunc expects lowercase strings
    )
    
    if len(columns)==0: columns=''
    cur_key=(index, columns, value, aggfunc)
    
    pt = pivot.iloc[:100,:100] 
    pr=Prompt(pt=pt, value=value, aggfunc=aggfunc, unique_dict = unique_dict)

    utility = Utility(pt, d_size=len(df), t_size=len(pt), original_table=df, llm=llm, 
                        pr=pr, query=comb, interesting_attributes=interesting_attrs,
                        aggfunc_ranks=aggfunc_ranks,
                        clf_trd=clf_trd, clf_out=clf_out,
                        gamma=gamma, tau_cor=tau_cor, tau_rat=tau_rat, output_dir=output_dir,
                        concise_cache=concise_cache, semantics_cache=semantics_cache,
                        )
    
    if prune: utility.compute_utility_score()
    else: utility.compute_utility_score_wo_pruning()
    
    score = utility.utility_score
    result = {}
    result["key"] = cur_key
    result["utility score"] = score
    result["density score"] = utility.int_model.density
    result["query semantics score"] = utility.int_model.query_semantics
    result["concise score"] = utility.int_model.concise
    result["interpretability score"] = utility.int_model.interpretability_score
    
    result["informativeness score"] = utility.ins_model.informativeness_score
    result["correlation score"] = utility.ins_model.correlation_score
    result["ratio score"] = utility.ins_model.ratio_score
    result["trend score"] = utility.ins_model.trend_score
    result["outlier score"] = utility.ins_model.surprise_score
    result["insightfulness score"] = utility.ins_model.insightfulness_score
    
    return result, pt, sql_query

def compute_utility_with_embeddings(
    combs: List[Dict[str, Any]],
    df: pd.DataFrame,
    llm,
    interesting_attrs: List[str],
    aggfunc_ranks: Dict[str, float],
    clf_trd, clf_out,
    args,
    output_dir: str,
    save_sql: bool = True,
    save_pivot: bool = False,
    verbose: bool = True,
    unique_dict:dict = None,
    gamma:float=5.0, tau_cor:float=0.7, tau_rat:float=5.0,
    prune:bool=True,
    do_parallel:bool = False,
    concise_cache:dict=None, semantics_cache:dict=None,
) -> Tuple[
    List[Dict[str, Any]],       # results
    List[Dict[str, str]],       # sql_queries
    List[Tuple[Tuple, pd.DataFrame]],  # pivots
]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    sql_queries = []
    pivots = []

    start_time = time.time()

    for comb in combs:
#         try:
        result, pivot, sql_query = get_single_utilty(
            comb, df, llm, interesting_attrs, aggfunc_ranks, clf_trd, clf_out, unique_dict, output_dir,
            gamma=gamma, tau_cor=tau_cor, tau_rat=tau_rat,
            prune=prune, concise_cache=concise_cache, semantics_cache=semantics_cache,
        )
        results.append(result)
        sql_queries.append(sql_query)
        pivots.append(pivot)
#         except Exception as e:
#             print(f"[Utility] Failed on comb: {comb} → {e}")
#             continue

    elapsed = time.time() - start_time

    if llm or not do_parallel:
        if results:
            keys = results[0].keys()
            csv_path = os.path.join(output_dir, "utility_scores.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

        with open(os.path.join(output_dir, "utility_total_time.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])  
            writer.writerow(["Total query elapsed time (seconds)", f"{elapsed:.4f}"])

        if verbose:
            print(f"[Utility] Saved {len(results)} utility results in {elapsed:.2f}s → {output_dir}/utility_total_time.csv")

    return results, sql_queries, pivots

def compute_utility(
    combs: List[Dict[str, Any]],
    df: pd.DataFrame,
    llm,
    interesting_attrs: List[str],
    aggfunc_ranks: Dict[str, float],
    clf_trd, clf_out,
    args,
    output_dir: str,
    save_sql: bool = True,
    save_pivot: bool = False,
    verbose: bool = True,
    unique_dict:dict = None,
    ) -> Tuple[
        List[Dict[str, Any]],       # results
    ]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    sql_queries = []
    pivots = []

    start_time = time.time()

    for comb in combs:
        try:
            result, pivot, sql_query = get_single_utilty(
                comb, df, llm, interesting_attrs, aggfunc_ranks, clf_trd, clf_out, unique_dict, output_dir
            )
            results.append(result)
            sql_queries.append(sql_query)
            pivots.append(pivot)
        except Exception as e:
            print(f"[Utility] Failed on comb: {comb} → {e}")
            continue

    elapsed = time.time() - start_time

    # Save utility scores
    if results:
        keys = results[0].keys()
        csv_path = os.path.join(output_dir, "utility_scores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    # if save_sql:
    #     with open(os.path.join(output_dir, "sql_queries.json"), "w") as f:
    #         json.dump(sql_queries, f, indent=2)

    # Save timing info
    with open(os.path.join(output_dir, "utility_total_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])  
        writer.writerow(["Total query elapsed time (seconds)", f"{elapsed:.4f}"])

    if verbose:
        print(f"[Utility] Saved {len(results)} utility results in {elapsed:.2f}s → {output_dir}/utility_total_time.csv")

    return results