# Asking LLM what could be 5 interesting pivot table in this data?
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
from collections import Counter
import re
import warnings
import torch
import pandas as pd
import argparse
import time 
import json 
import gc

import torch

import time
import json
import csv
import argparse
import torch
import pandas as pd
from src.preprocessing import Preprocess
from src.helper import *
from src.utility.llama import Llama3

class Llama3:
    def __init__(self, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
#     def __init__(self, model_path="tablegpt/TableGPT2-7B"):
        """
        Initializes the LLM with given model_path

        Args:
            model_path (str, optional): We use Meta-Llama-3-8B-Instruct to utilize chat-like result. 
                                        Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
            batch_size=1,
        )
        self.pipeline.model.config.pad_token_id = self.pipeline.model.config.eos_token_id
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]
  
    def get_response(
          self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id = self.pipeline.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][len(prompt):]

        print(prompt)
        print(response)
        
        del prompt 
        del outputs

        return response
    
    def chatbot(self, query, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        user_input = "User: "+query
        response = self.get_response(user_input, conversation)
        return response

    def get_pivot_tables_prompt(self, k):
        instruction=None
        if k==1:        
            instruction = f"""Given a table, please identify interesting exact one pivot table. Please use at most one column for each parameter to create the pivot table with aggregation functions only from mean, count, sum, min, and max. Do not use median, mode, std, etc. The table can be insightful, such as having high deviation, correlation, or outliers. Also please consider interpretability as well, such as the size of the table, the appropriateness of the aggregate function for the value attribute, and the attribute semantics. """
            
        else: 
            instruction =  f"""Given a table, please identify interesting and diverse exact {k} pivot tables. Please use at most one attributes for each parameter of index, column, and value to create the pivot table with aggregation functions only from mean, count, sum, min, and max. Do not use median, mode, std, etc. You can empty the column parameter if you want. The table can be insightful, such as having high deviation, correlation, or outliers. Also please consider interpretability as well, such as the size of the table, the appropriateness of the aggregate function for the value attribute, and the attribute semantics. """
        
        return instruction
    
# This is the translation from dataframe to the string for generative language model.
def dataframe_to_str(df):
    # Generate the header row
    headers = list(df.columns.astype(str))
    header_row = "| " + " | ".join(headers) + " |"
    
    # Generate the separator row
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Generate the data rows
    data_rows = []
    for row in df.itertuples(index=False, name=None):
        data_rows.append("| " + " | ".join(map(str, row)) + " |")
    
    # Combine all parts into a single string
    str_values = "\n".join([header_row, separator_row] + data_rows)
    
    return str_values

# You need to recommend interesting data trends, such as high variance or correlation, and provide the reasons why you chose the tables.
def generate_message_interesting_pivot_tables(instruction:str, table_str: str, k: int):
    prompt = None
    if k == 1:
        prompt="Task: "+instruction + f""" Return one pivot table as JSON in the format as {{pivot_table:{{pivot_table_1: {{index: {{column header}}, column: {{column header}}, value: {{column header}}, agg: {{aggregate function}}}}}}}}. Please just return JSON result. 
        [Q]: **Table Data:** 

            Input:
            """ + table_str+ """

            Return the pivot table as JSON in the format as
            { pivot_table:
            pivot_table_1: {index: {attribute name}, column: {attribute name}, value: {attribute name}, agg: {attribute name}} 
            }. Please just return JSON result.

        [A]:"""
    else:
        prompt="Task: "+instruction + f""" Return {k} pivot tables as JSON in the format as {{pivot_tables:{{pivot_table_1: {{index: {{column header}}, value: {{attribute name}}, column: {{attribute name}}, agg: {{aggregate function}}}}, pivot_table_2: {{index: {{column header}}, value: {{attribute name}}, column: {{attribute name}}, agg: {{aggregate function}}}}, ...}}}} with the result values of pivot tables.
        [Q]: **Table Data:** 

            Input:
            """ + table_str+ """

            Return the pivot tables as JSON in the format as
            { pivot_tables:
            pivot_table_1: {index: {attribute name}, value: {attribute name}, column: {attribute name}, agg: {aggregate function}},
            pivot_table_2: {index: {attribute name}, value: {attribute name}, column: {attribute name}, agg: {aggregate function}}, 
            ...
            } and the result values of pivot tables.  

        [A]:"""
        
    return prompt

def paraphrase_instruction(instruction: str, bot):
    prompt="""Please paraphrase following sentence. Return the final result as JSON in the format as {"paraphrased_sentence": "result of paraphrasing"}.: """\
    + instruction
    
    response=bot.chatbot(prompt)
    agg = find_value_in_response(response, "paraphrased_sentence")
    return prompt, agg

def find_value_in_response(response, key):
    key_idx = response.find(f'"{key}"')

    # Key not found
    if key_idx == -1:
        return None  

    start_idx = response.rfind('{', 0, key_idx)
    end_idx = response.find('}', key_idx)
  
    # Invalid JSON structure around the key
    if start_idx == -1 or end_idx == -1:
        return None  

    # Extract the JSON-like substring
    json_str = response[start_idx:end_idx+1]

    # Parse the JSON string into a Python dictionary
    try:
        data = json.loads(json_str)
        return data.get(key)
    except:
        return None
    return data[key]

def find_value_in_final_response(response, key):
    print(response)
    
    json_part = re.search(r'\{.*\}', response, re.DOTALL).group(0)

    try:
        data = json.loads(json_part)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

    pivot_tables = data.get(key, [])
    
    if isinstance(pivot_tables, list):
        return pivot_tables
    elif isinstance(pivot_tables, dict):
        return list(pivot_tables.values())
    else:
        raise ValueError("Invalid structure for 'pivot_tables'. Must be a list or dict.")

def get_interesting_pivot_tables(data, bot, k):
    instruction_1= bot.get_pivot_tables_prompt(k)
    instructions = [instruction_1]
    table_str=dataframe_to_str(data)
    results=[]
    for idx, inst in enumerate(instructions):
        prompt=generate_message_interesting_pivot_tables(inst, table_str, k)
        print(prompt)
        response=bot.chatbot(prompt)
        if k==1:
            comb = find_value_in_final_response(response, "pivot_table")
        else:
            comb = find_value_in_final_response(response, "pivot_tables")
        results = comb
    
#     filtered_results=majority_voting(results)
#     print(filtered_results)
    return results, results

def majority_voting(data: list):
    flattened_data = []
    for sublist in data:
        for item in sublist:
            flattened_data.append(item)

    counter = Counter(flattened_data)

    # threshold = len(data) / 2
    threshold = 0

    majority_vote_items = [item for item, count in counter.items() if count > threshold]
    
    return majority_vote_items

if __name__=="__main__":
    
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Experiment pipeline with argument control")
    parser.add_argument('--filename', type=str, default='marketing_data', help='CSV file name')
    parser.add_argument('--filepath', type=str, default='..', help='CSV file location')
    parser.add_argument('--transformer_path', type=str, default='/scratch/general/nfs1/u1472329/cache/', help='Transformer model location')
    parser.add_argument('--hf_token_path', type=str, default='hf_token.txt', help='Huggingface token path for llama and tapex')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--output_dir', type=str, default='result/greedy/standardscale/', help='output directory')
    parser.add_argument("--k", type=int, default=10, help="The pivot table budgets", required=True)

    # Parameters
    parser.add_argument('--data_num', type=float, default=0.2, help='The number/fraction of rows of data')
    parser.add_argument('--max_columns', type=int, default=None, help='Maximum number of columns to use from the DataFrame')
    args = parser.parse_args()

    set_environment(args.transformer_path, args.hf_token_path)
    set_random_seed(args.seed)

    filename = f"{args.filepath}{args.filename}.csv"
    df = pd.read_csv(filename, index_col=False).dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].fillna(0)
    
    if args.max_columns is not None and args.max_columns < df.shape[1]:
        df = df.iloc[:, :args.max_columns]
    df = Preprocess(df).do_preprocess()
    
    if args.data_num <= 1:
            df = df.sample(frac=args.data_num, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=int(args.data_num), random_state=42).reset_index(drop=True)
        
    bot = Llama3()

    k=args.k
    
    params = set_params(args)
    prune_dir, utility_dir, diversity_dir = get_experiment_folders(args.output_dir, params)

    output={}
    cnt=0

    start_time = time.time()
    _, best_set = get_interesting_pivot_tables(df, bot, k)
    end_time = time.time()
    
    best_combs = []

    for comb in best_set:
        sorted_comb = {
            "index": min(comb.get("index", None), comb.get("column", '') if len(comb.get("column", '')) != 0 else ''),
            "column": max(comb.get("index", None), comb.get("column", '') if len(comb.get("column", '')) != 0 else ''),
            "value": comb.get("value", None),
            "agg": comb.get("agg", '').lower()
        }

        best_combs.append(sorted_comb)

    for comb in best_set:
        transposed_comb = {
            "index": min(comb.get("column", '') if len(comb.get("column", '')) != 0 else '',
                          comb.get("index", None)),
            "column": max(comb.get("column", '') if len(comb.get("column", '')) != 0 else '',
                          comb.get("index", None)),
            "value": comb.get("value", None),
            "agg": comb.get("agg", '').lower()
        }
        best_combs.append(transposed_comb)

#     final_best_combs = [dict(item) for item in best_combs]
#     best_combs = final_best_combs
    
    best_set = {
        (
            comb.get("index", None),
            comb.get("column", ''),
            comb.get("value", None),
            comb.get("agg", '').lower()
        )
    for comb in best_combs
    }
    
    print(f"best set {best_set}") 
    llm_execution_time = end_time - start_time
    print(f"LLM Calculation time: {llm_execution_time:.6f} seconds")
    
    os.makedirs(diversity_dir, exist_ok=True)

    # 1. Save best_set
    best_set_path = os.path.join(diversity_dir, "best_set.csv")
    with open(best_set_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "column", "value", "aggfunc"])  
        for entry in best_set:
            writer.writerow(entry)

    print(f"Saved best_set to {best_set_path}")

    time_path = os.path.join(diversity_dir, "time.csv")
    with open(time_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["llm_execution_time_seconds"])
        writer.writerow([f"{llm_execution_time:.6f}"])

    print(f"Saved execution time to {time_path}")
