import argparse
import torch
import pandas as pd
from src.preprocessing import Preprocess
from src.generate_table import generate_canonical_combinations
from src.helper import *
from src.utility.llama import Prompt, Llama3
from src.utility.utility import get_precomputing_result
import pandas as pd
import numpy as np
import os 

import torch.nn as nn
import torch.nn.init as init
from functools import partial
from multiprocessing import Pool, cpu_count
import csv 
import time

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
            
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        
    def forward(self, x):
        score = self.model(x)
        return score

def normalize_values(values):
    if not values.size == 0:
        min_val = np.min(values)
        max_val = np.max(values)
    else:
        min_val = 0.0
        max_val = 0.0
    return (values - min_val) / (max_val - min_val) if max_val != min_val else values

def extract_features(df: pd.DataFrame):
    tuple_count = int(len(df)) 
    
    numeric_df = df.select_dtypes(include='number')
    
    value_sum = float(numeric_df.sum().sum())
    
    if not numeric_df.empty:
        max_min = float(numeric_df.max().max() - numeric_df.min().min())
    else:
        max_min = 0.0 
    
    try:
        s_dev = float(numeric_df.stack().std().item())
    except:
        s_dev = 0.0  
    
    normalized_values = normalize_values(numeric_df.stack().values) 
    top_10_values = np.sort(normalized_values)[-10:]  
    top_10_percentages = [float(np.sum(normalized_values == v) / len(normalized_values)) for v in top_10_values]  

    while len(top_10_percentages) < 10:
        top_10_percentages.append(0.0)
        
    features = {
        "tuple_count": tuple_count,
        "value_sum": value_sum,
        "max_min": max_min,
        "s_dev": s_dev,
        "top_10_percentages": top_10_percentages  
    }
    
    flat_vector = np.array([
    features["tuple_count"],
    features["value_sum"],
    features["max_min"],
    features["s_dev"],
    *features["top_10_percentages"]  
    ], dtype=np.float32)

    input_tensor = torch.from_numpy(flat_vector).unsqueeze(0)

    return input_tensor

def load_mlp_model(model_path, model_class):
    model = model_class(14)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_pivot_table(df, comb):
    index = comb["index"]
    columns = comb["column"]
    value = comb["value"]
    aggfunc = comb["aggfunc"].upper()
    
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        columns = []
            
    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=value,
        aggfunc=aggfunc.lower() 
    )
    pivot = pivot.iloc[:100,:100] 
    
    return pivot
def make_mlp_model():
    return MLP(input_size=14)
def score_combination(comb, df, model_state_dict, model_factory):
    model = model_factory()
    model.load_state_dict(model_state_dict)
    model.eval()

    pivot_df = generate_pivot_table(df, comb)
    input_tensor = extract_features(pivot_df)

    with torch.no_grad():
        score = model(input_tensor).item()

    return (comb, score)

def evaluate_combinations_parallel(pruned_result, df, model, model_factory, top_k=20, num_workers=12):
    model_state_dict = model.state_dict()
    total = len(pruned_result)
    chunksize = max(1, total // (10 * num_workers)) 

    with Pool(processes=num_workers) as pool:
        func = partial(score_combination, df=df, model_state_dict=model_state_dict, model_factory=model_factory)
        scored_results = pool.map(func, pruned_result, chunksize=chunksize)

    top_combs = sorted(scored_results, key=lambda x: x[1], reverse=True)[:top_k]
    return top_combs

def evaluate_combinations(pruned_result, df, model, top_k=20):
    scored_results = []

    for comb in pruned_result:
        pivot_df = generate_pivot_table(df, comb)  
        features = extract_features(pivot_df)       

        input_tensor = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = model(input_tensor).item()

        scored_results.append((comb, score))

    top_combs = sorted(scored_results, key=lambda x: x[1], reverse=True)[:top_k]
    return top_combs

def flatten_comb_dict(comb_dict):
    flat = {}
    for k, v in comb_dict.items():
        if isinstance(v, list):
            flat[k] = ",".join(map(str, v))  # Convert list to comma-separated string
        else:
            flat[k] = v
    return flat

def save_top_results(top_combs, output_dir, filename="top_20_combinations.csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Prepare rows
    rows = []
    for comb, score in top_combs:
        flat_comb = flatten_comb_dict(comb)
        flat_comb["score"] = score  # Add score
        rows.append(flat_comb)

    # Write to CSV
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Top-{len(top_combs)} combinations saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment pipeline with argument control")
    parser.add_argument('--filename', type=str, default='marketing_data', help='CSV file name')
    parser.add_argument('--filepath', type=str, default='..', help='CSV file location')
    parser.add_argument('--model_path', type=str, default='./daisy.pth', help='daisy location')
    parser.add_argument('--transformer_path', type=str, default='/scratch/general/nfs1/u1472329/cache/', help='Transformer model location')
    parser.add_argument('--hf_token_path', type=str, default='hf_token.txt', help='Huggingface token path for llama and tapex')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--clf_trd_path', type=str, default='prompt_learning/model/marketing_data_10000_decision_tree.pkl', help='for trend prompt cache path')
    parser.add_argument('--clf_out_path', type=str, default='prompt_learning/model/marketing_data_10000_outlier_decision_tree.pkl', help='for outlier prompt cache path')
    
    parser.add_argument('--gamma', type=float, default=5000.0, help='Factor for informativeness')
    parser.add_argument('--int_threshold', type=float, default=0.5, help='Interpretability threshold')
    parser.add_argument('--threshold', type=float, default=0.5, help='Diversity threshold')
    parser.add_argument('--k', type=int, default=5, help='Number of k-centers')
    parser.add_argument('--df_sample_num', type=int, default=5, help='Sample size for precomputing')
    parser.add_argument('--data_num', type=float, default=0.2, help='The number/fraction of rows of data')
    parser.add_argument('--max_columns', type=int, default=None, help='Maximum number of columns to use from the DataFrame')

    parser.add_argument('--tau_cor', type=float, default=0.7, help='for correlation tau')
    parser.add_argument('--tau_rat', type=float, default=5.0, help='for ratio tau')
    
    parser.add_argument('--output_dir', type=str, default='result/daisy/', help='output directory')

    parser.add_argument('--do_prune', type=str2bool, default=True, help='Do you want to do prune?')
    
    parser.add_argument('--do_timeout', type=str2bool, default=False, help='Do you want timeout?')
    parser.add_argument('--timeout', type=int, default=4800, help='Set the time out')

    return parser.parse_args()

def main(args):
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
    combs = random.sample(combs, len(combs))

    params = set_params(args)
    prune_dir, utility_dir, diversity_dir = get_experiment_folders(args.output_dir, params)

    if args.data_num <= 1:
            df = df.sample(frac=args.data_num, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=int(args.data_num), random_state=42).reset_index(drop=True)
        
    model = load_mlp_model(args.model_path, MLP) 
    print("Starting combination evaluation...")
    start_time = time.time() # Record the start time

    top_combs = evaluate_combinations_parallel(
        combs,
        df,
        model,
        model_factory=make_mlp_model,
        top_k=20,
        num_workers=12
    )

    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time # Calculate the duration

    print(f"Combination evaluation finished.")
    print(f"Time taken for evaluate_combinations_parallel: {elapsed_time:.2f} seconds")

    save_top_results(top_combs, diversity_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)