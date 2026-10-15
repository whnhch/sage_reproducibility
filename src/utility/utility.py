from src.utility.insightfulness import *
from src.utility.interpretability import *
from typing import List, Dict, Any, Tuple
import os
import time
import csv

def get_precomputing_result(df:pd.DataFrame, df_sample_num=5, llm:Llama3=None, pr:Prompt=None):
    interesting_attrs= get_interesting_attributes(llm=llm,pr=pr,df=df.sample(n=10, random_state=42).reset_index(drop=True))
    unique_values_dict = retrieve_unique_values_from_df(df, interesting_attrs)
    attribute_aggfunc_ranks = get_ranked_aggfunc(llm=llm,pr=pr,df=df.sample(n=10, random_state=42).reset_index(drop=True))
    
    return interesting_attrs, unique_values_dict, attribute_aggfunc_ranks

def get_precomputing_saved_result(df:pd.DataFrame, interesting_attrs_path, attribute_aggfunc_ranks_path):
    def read(path: Path):
        with open(path, "r") as f:
            return json.load(f)
    interesting_attrs= read(interesting_attrs_path)
    unique_values_dict = retrieve_unique_values_from_df(df, interesting_attrs)
    attribute_aggfunc_ranks = read(attribute_aggfunc_ranks_path)
    
    return interesting_attrs, unique_values_dict, attribute_aggfunc_ranks

def get_interesting_attributes(llm: Llama3=None, pr:Prompt=None, df:pd.DataFrame=None):
    vars = pr.prepare_interesting_column(df)
    instruction = pr.load_instruction('interesting_columns', vars)
    result = llm.get_completion(instruction=instruction, pr=pr, 
                                task_name='interesting_columns', 
                                do_paraphrase=True, keywords=vars)
    return result

def get_ranked_aggfunc(llm: Llama3=None, pr:Prompt=None, df:pd.DataFrame=None):
    results = {}

    for col in df.columns:
        vars = pr.prepare_interesting_aggfunc(df[col], col)
        instruction = pr.load_instruction('interesting_aggfunc', vars)
        result = llm.get_completion(instruction=instruction, pr=pr, 
                                    task_name='interesting_aggfunc', 
                                do_paraphrase=False, keywords=vars)
        results[col] = result
        
    return results

def indicator(query: dict, interesting_attrs: list[str]) -> float:
    """
    Returns 1.0 if ALL of index, column, and value are in interesting attributes.
    Otherwise returns 0.0.

    Args:
        query: A dictionary with keys 'index', 'column', 'value', 'aggfunc'.
        interesting_attrs: A list of interesting attribute names.

    Returns:
        1.0 if all parts of the query are interesting; otherwise 0.0.
    """
    keys = ['index', 'column', 'value']
    for key in keys:
        value = query.get(key)
        if isinstance(value, (list, tuple)):
            if not all(v in interesting_attrs for v in value):
                return False
        else:
            if value not in interesting_attrs:
                return False
    return True

def retrieve_unique_values(attr: pd.Series, max_unique: int = 100, bin_numeric: bool = True, bin_count: int = 5) -> List[Any]:
    attr = attr.dropna()
    unique_vals = attr.unique()
    
    if len(unique_vals) > max_unique:
        return []

    if bin_numeric and pd.api.types.is_numeric_dtype(attr) and len(unique_vals) > 10:
        try:
            #  bucketize the numerics
            binned = pd.qcut(attr, q=bin_count, duplicates='drop')
            return binned.astype(str).unique().tolist()
        except ValueError:
            return unique_vals.tolist()
    
    return unique_vals.tolist()

def retrieve_unique_values_from_df(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[Any]]:
    return {
        col: retrieve_unique_values(df[col])
        for col in columns
        if col in df.columns
    }

class Utility:
    def __init__(self, pt, d_size=None, t_size=100, original_table=None, query:dict = None, 
                 llm: Llama3=None, pr:Prompt=None, 
                 interesting_attributes:List=None,
                 aggfunc_ranks:dict=None,
                 clf_trd=None, clf_out=None,
                 gamma:float=5000.0, tau_cor:float=0.7, tau_rat:float=5.0, output_dir:str=None,
                 concise_cache:dict=None, semantics_cache:dict=None,
                 ):
        # df : NxM
        unique_values_dict = retrieve_unique_values_from_df(original_table, interesting_attributes)
        
        
        # self.ins_model = ImprovedInsightfulness(pt, d_size=d_size, t_size=t_size, 
        self.ins_model = NaiveInsightfulness(pt, d_size=d_size, t_size=t_size, 
                                             original_table=original_table, query=query, 
                                             llm=llm, pr=pr, 
                                             clf_trd=clf_trd, clf_out=clf_out,
                                             gamma=gamma, tau_cor=tau_cor, tau_rat=tau_rat)
        self.int_model = NaiveInterpretability(pt, d_size=d_size, t_size=t_size, original_table=original_table, query=query, 
                                               llm=llm, pr=pr, 
                                               unique_dict = unique_values_dict, aggfunc_ranks=aggfunc_ranks,
                                               concise_cache=concise_cache, semantics_cache=semantics_cache,
                                               )
        
        # row_frac = 1  # sample 20% of rows
        # col_frac = 1  # sample 20% of columns

        # n_rows = max(1, int(pt.shape[0] * row_frac))
        # n_cols = max(1, int(pt.shape[1] * col_frac))

        # sampled = pt.sample(n=n_rows, random_state=42)
        # sampled = sampled.sample(n=n_cols, axis=1, random_state=42)
        self.pt = pt
        
        self.pt = pt
        self.query = query 
        
        self.interpretability=0
        self.insightfulness=0
        self.interesting_columns = False
        
        self.alpha = 0.5
        self.utility_score = 0
        self.interesting_columns = indicator(query, interesting_attributes)
        
        self.output_dir = output_dir
            
    def compute_utility_score(self):
        if self.interesting_columns:
            start_ins = time.time()
            self.ins_model.calculate_scores(self.pt)
            end_ins = time.time()
            insight_time = end_ins - start_ins
            self.insightfulness = self.ins_model.insightfulness_score
        else:
            insight_time = 0
            self.insightfulness = 0
            
        start_int = time.time()
        self.int_model.calculate_scores()
        end_int = time.time()
        interpret_time = end_int - start_int
        self.interpretability = self.int_model.interpretability_score

        self.utility_score = self.alpha * self.insightfulness + (1 - self.alpha) * self.interpretability

        # query_repr = self.query

        # csv_path = os.path.join(self.output_dir, "utility_score_time.csv")
        # file_exists = os.path.exists(csv_path)

        # with open(csv_path, "a", newline="") as f:
            # writer = csv.writer(f)
            # if not file_exists:
                # writer.writerow(["Query", "Metric", "Elapsed Time (seconds)"])

            # writer.writerow([query_repr, "Insightfulness", f"{insight_time:.4f}"])
            # writer.writerow([query_repr, "Interpretability", f"{interpret_time:.4f}"])

    def compute_utility_score_wo_pruning(self):
        if self.interesting_columns:
            start_ins = time.time()
            self.ins_model.calculate_scores(self.pt)
            end_ins = time.time()
            insight_time = end_ins - start_ins
            self.insightfulness = self.ins_model.insightfulness_score

        else:
            insight_time = 0
            self.insightfulness = 0
            
        start_int = time.time()
        self.int_model.calculate_scores()
        end_int = time.time()
        interpret_time = end_int - start_int
        self.interpretability = self.int_model.interpretability_score

        self.utility_score = self.alpha * self.insightfulness + (1 - self.alpha) * self.interpretability

        # query_repr = self.query

        # csv_path = os.path.join(self.output_dir, "utility_score_time.csv")
        # file_exists = os.path.exists(csv_path)

        # with open(csv_path, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     if not file_exists:
        #         writer.writerow(["Query", "Metric", "Elapsed Time (seconds)"])

        #     writer.writerow([query_repr, "Insightfulness", f"{insight_time:.4f}"])
        #     writer.writerow([query_repr, "Interpretability", f"{interpret_time:.4f}"])