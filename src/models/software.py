'''
Get inputs of list of results, and return the scores.
'''
import argparse
import numpy as np
import torch
import pandas as pd

from src.preprocessing import Preprocess
from src.generate_table import generate_canonical_combinations
from src.diversity.helper import concat_embeddings, get_all_embeddings, get_embeddings, greedy_selection
from src.helper import *
from src.utility.helper import compute_utility_with_embeddings, parallel_compute_utility
from src.utility.llama import Prompt, Llama3
from src.utility.utility import get_precomputing_result, get_precomputing_saved_result
from scipy.spatial.distance import cdist

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment pipeline with argument control")
    parser.add_argument('--combname', type=str, default='marketing_data', help='CSV file name')
    parser.add_argument('--combpath', type=str, default='..', help='CSV file location')
    parser.add_argument('--filename', type=str, default='marketing_data', help='CSV file name')
    parser.add_argument('--filepath', type=str, default='..', help='CSV file location')
    parser.add_argument('--transformer_path', type=str, default='/scratch/general/nfs1/u1472329/cache/', help='Transformer model location')
    parser.add_argument('--hf_token_path', type=str, default='hf_token.txt', help='Huggingface token path for llama and tapex')
    
    parser.add_argument('--data_num', type=float, default=0.2, help='The number/fraction of rows of data')
    parser.add_argument('--max_columns', type=int, default=None, help='Maximum number of columns to use from the DataFrame')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k', type=int, default=3, help='k')
    
    parser.add_argument('--gamma', type=float, default=5000.0, help='Factor for informativeness')
    parser.add_argument('--df_sample_num', type=int, default=5, help='Sample size for precomputing')
    parser.add_argument('--clf_trd_path', type=str, default='prompt_learning/model/marketing_data_10000_decision_tree.pkl', help='for trend prompt cache path')
    parser.add_argument('--clf_out_path', type=str, default='prompt_learning/model/marketing_data_10000_outlier_decision_tree.pkl', help='for outlier prompt cache path')
    
    parser.add_argument('--tau_cor', type=float, default=0.5, help='for correlation tau')
    parser.add_argument('--tau_rat', type=float, default=2.0, help='for ratio tau')
    
    parser.add_argument('--output_dir', type=str, default='result/software/', help='output directory')

    parser.add_argument('--interesting_attributes_path', type=str, default='./tmp', help='for interesting_attributes_path')
    parser.add_argument('--attribute_aggfunc_ranks', type=str, default='./tmp', help='for attribute_aggfunc_ranks')

    return parser.parse_args()

def main(args):
    set_environment(args.transformer_path, args.hf_token_path)
    set_random_seed(args.seed)

    filename = f"{args.filepath}{args.filename}.csv"
    df = pd.read_csv(filename, index_col=False).dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].fillna(0)
    df = Preprocess(df).do_preprocess()

    software_filename = f"{args.combpath}{args.combname}.csv"
    software_combs = pd.read_csv(software_filename, index_col=False).dropna(axis=1, how='all')
    software_combs["aggfunc"] = software_combs["aggfunc"].str.lower()
    software_combs["column"] = software_combs["column"].apply(lambda x: [] if x == '[]' else x)
    software_combs["value"] = software_combs["value"].apply(
        lambda x: [] if x.strip() == ''
        else x.strip() if ',' not in x
        else [item.strip() for item in x.split(',')]
    )
    software_combs = software_combs.to_dict(orient="records")

    pr = Prompt()
    llm = Llama3()

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

    clf_trd, clf_out = load_classifiers(args.clf_trd_path, args.clf_out_path)
    
    if args.data_num <= 1:
            df = df.sample(frac=args.data_num, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=int(args.data_num), random_state=42).reset_index(drop=True)
        
    def convert_columns_to_numeric(df):
        for col in df.columns:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only overwrite if at least one value was successfully converted
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                continue
        return df

    df = convert_columns_to_numeric(df)

    if clf_trd and clf_out: llm = None
    utility_results, sql_queries, pivot_tables = parallel_compute_utility(
        combs=software_combs,
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
        prune=False,
        num_workers=12,
        do_parallel=False,
    )
    
    query_checkpoint = 'gaussalgo/T5-LM-Large-text2sql-spider'
    table_checkpoint = 'microsoft/tapex-large'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_tokenizer, query_model, table_tokenizer, table_model = load_encoders(query_checkpoint, table_checkpoint, device)
    query_np, table_np = get_all_embeddings(sql_queries, pivot_tables, query_tokenizer, query_model, table_tokenizer, table_model, device)

    def save_cdist(matrix: np.ndarray, name: str):
        dist = cdist(matrix, matrix, metric="cosine")/2.0
        df = pd.DataFrame(dist)
        df.to_csv(os.path.join(diversity_dir, f"{name}_cosine_distance.csv"), index=False)

    def save_sqrt_cdist(matrix: np.ndarray, name: str):
        dist = cdist(matrix, matrix, metric="cosine")/2.0
        dist = np.clip(dist, 0, None)
        dist = np.sqrt(dist)
        df = pd.DataFrame(dist)
        df.to_csv(os.path.join(diversity_dir, f"{name}_cosine_distance.csv"), index=False)


    combined = np.hstack([query_np, table_np])

    save_sqrt_cdist(combined, "combined_sqrt")

if __name__ == "__main__":
    args = parse_args()
    main(args)