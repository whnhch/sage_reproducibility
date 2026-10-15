import argparse
import torch
import pandas as pd

from src.preprocessing import Preprocess
from src.generate_table import generate_canonical_combinations
from src.diversity.helper import concat_embeddings, compute_and_concat_embeddings_with_timing, greedy_selection
from src.helper import *
from src.utility.helper import parallel_compute_utility
from src.utility.llama import Prompt, Llama3
from src.utility.utility import get_precomputing_result, get_precomputing_saved_result
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment pipeline with argument control")
    parser.add_argument('--filename', type=str, default='marketing_data', help='CSV file name')
    parser.add_argument('--filepath', type=str, default='..', help='CSV file location')
    parser.add_argument('--transformer_path', type=str, default='/scratch/general/nfs1/u1472329/cache/', help='Transformer model location')
    parser.add_argument('--hf_token_path', type=str, default='hf_token.txt', help='Huggingface token path for llama and tapex')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--clf_trd_path', type=str, default='', help='for trend prompt cache path')
    parser.add_argument('--clf_out_path', type=str, default='', help='for outlier prompt cache path')
    
    parser.add_argument('--gamma', type=float, default=5000.0, help='Factor for informativeness')
    parser.add_argument('--int_threshold', type=float, default=0.5, help='Interpretability threshold')
    parser.add_argument('--threshold', type=float, default=0.5, help='Diversity threshold')
    parser.add_argument('--k', type=int, default=5, help='Number of k-centers')
    parser.add_argument('--df_sample_num', type=int, default=5, help='Sample size for precomputing')
    parser.add_argument('--data_num', type=float, default=0.2, help='The number/fraction of rows of data')
    parser.add_argument('--max_columns', type=int, default=None, help='Maximum number of columns to use from the DataFrame')

    parser.add_argument('--tau_cor', type=float, default=0.7, help='for correlation tau')
    parser.add_argument('--tau_rat', type=float, default=5.0, help='for ratio tau')
    
    parser.add_argument('--output_dir', type=str, default='result/greedy/standardscale/', help='output directory')

    parser.add_argument('--do_prune', type=str2bool, default=True, help='Do you want to do prune?')
    
    parser.add_argument('--do_timeout', type=str2bool, default=False, help='Do you want timeout?')
    parser.add_argument('--timeout', type=int, default=4800, help='Set the time out')
    parser.add_argument('--do_parallel', type=str2bool, default=True, help='Do you want to do parallel computation?')
    parser.add_argument('--do_cache', type=str2bool, default=True, help='Do you want to do cache computation?')

    parser.add_argument('--do_int_cache', type=str2bool, default=True, help='Do you want to do interpretability cache computation?')

    parser.add_argument('--interesting_attributes_path', type=str, default='./tmp', help='for interesting_attributes_path')
    parser.add_argument('--attribute_aggfunc_ranks', type=str, default='./tmp', help='for attribute_aggfunc_ranks')

    return parser.parse_args()

def main(args):
    print(sys.executable)

    set_environment(args.transformer_path, args.hf_token_path)
    set_random_seed(args.seed)

    filename = f"{args.filepath}{args.filename}.csv"
    df = pd.read_csv(filename, index_col=False).dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].fillna(0)
    
    if args.max_columns is not None and args.max_columns < df.shape[1]:
        df = df.iloc[:, :args.max_columns]
    df = Preprocess(df).do_preprocess()

    result = generate_canonical_combinations(df, df.columns)
    combs = result[0] + result[1]

    pr = Prompt()
    llm = Llama3()

    clf_trd, clf_out = None, None
    if args.clf_trd_path != '': clf_trd, clf_out = load_classifiers(args.clf_trd_path, args.clf_out_path)
    
    query_checkpoint = 'gaussalgo/T5-LM-Large-text2sql-spider'
    table_checkpoint = 'microsoft/tapex-large'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_tokenizer, query_model, table_tokenizer, table_model = load_encoders(query_checkpoint, table_checkpoint, device)
    
    if os.path.exists(args.interesting_attributes_path) and os.path.exists(args.attribute_aggfunc_ranks):
        print("Using existing files for attributes")
        interesting_attrs, unique_values_dict, aggfunc_ranks = get_precomputing_saved_result(
            df=df,
            interesting_attrs_path=args.interesting_attributes_path,
            attribute_aggfunc_ranks_path=args.attribute_aggfunc_ranks
        )
    else:
        interesting_attrs, unique_values_dict, aggfunc_ranks = get_precomputing_result(
            df=df,
            df_sample_num=args.df_sample_num,
            llm=llm,
            pr=pr
        )
    params = set_params(args)

    prune_dir, utility_dir, diversity_dir = get_experiment_folders(args.output_dir, params)

    if args.do_prune:
        # if len(combs) < 10000 or not args.do_parallel:
        pruned_result = prune_combinations(
        combs=combs,
        interesting_attrs=interesting_attrs,
        df=df,
        unique_values_dict=unique_values_dict,
        aggfunc_ranks=aggfunc_ranks,
        int_threshold=args.int_threshold,
        output_dir=prune_dir,
        do_int_cache=args.do_int_cache,
        )
        # else:
        #     pruned_result = prune_combinations_parallel(
        #     combs=combs,
        #     interesting_attrs=interesting_attrs,
        #     df=df,
        #     unique_values_dict=unique_values_dict,
        #     aggfunc_ranks=aggfunc_ranks,
        #     int_threshold=args.int_threshold,
        #     output_dir=prune_dir,
        #     num_workers=12,
        #     )
    else:
        pruned_result = combs 

    concise_cache = None
    semantics_cache = None
    # pruned_result may be a plain list (legacy) or a dict with caches (when return_cache=True)
    if isinstance(pruned_result, list):
        pass
    else:
        pruned_result_ = pruned_result.get("pruned", [])
        concise_cache = pruned_result.get("concise_cache")
        semantics_cache = pruned_result.get("semantics_cache")
        pruned_result = pruned_result_
    if not args.do_int_cache: concise_cache, semantics_cache = None, None

    # let's use the fraction of the df
    # if args.data_num <= 1:
    #         df = df.sample(frac=args.data_num, random_state=42).reset_index(drop=True)
    # else:
    #     df = df.sample(n=int(args.data_num), random_state=42).reset_index(drop=True)
    if args.data_num <= 1:
        df = df.iloc[:int(len(df) * args.data_num)].reset_index(drop=True)
    else:
        df = df.iloc[:int(args.data_num)].reset_index(drop=True)
    
    if not args.do_cache: clf_trd , clf_out = None, None
    elif clf_trd and clf_out: llm = None
    utility_results, sql_queries, pivot_tables = parallel_compute_utility(
        combs=pruned_result,
        df=df,
        llm=llm, 
        interesting_attrs=interesting_attrs,
        aggfunc_ranks=aggfunc_ranks,
        clf_trd=clf_trd,
        clf_out=clf_out,
        args_obj=args,
        output_dir=utility_dir,
        unique_dict=unique_values_dict,
        gamma=args.gamma,
        tau_cor=args.tau_cor,
        tau_rat=args.tau_rat,
        prune=args.do_prune,
        num_workers=12,  
        do_parallel = args.do_parallel,
        concise_cache=concise_cache, semantics_cache=semantics_cache,
    )

    combined = compute_and_concat_embeddings_with_timing(
        sql_queries, pivot_tables,
        query_tokenizer, query_model,
        table_tokenizer, table_model,
        device,
        32,
        output_dir=diversity_dir,
        concat_embeddings_fn=concat_embeddings
    )

    greedy_selection(pruned_result, utility_results, combined, args.threshold, args.k, diversity_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)