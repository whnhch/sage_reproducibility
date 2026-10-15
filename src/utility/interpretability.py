import pandas as pd
import numpy as np
from typing import List
from src.utility.llama import *
from typing import List, Any, Union, Optional

RANK_MAP={
    0: 1.0,
    1: 0.8,
    2: 0.6,
    3: 0.4,
    4: 0.2,
}
class NaiveInterpretability:
    def __init__(self, pt=None, d_size=None, t_size=100, original_table=None, query=None, llm: Llama3=None, pr:Prompt=None,
                unique_dict: dict =None, aggfunc_ranks: dict = None, prune: bool = False,
                concise_cache: dict = None, semantics_cache: dict = None):
        # df : NxM
        self.d_size = d_size
        self.t_size = t_size
        
        self.original_table = original_table
        self.query = query
        self.unique_dict = unique_dict
        self.aggfunc_ranks = aggfunc_ranks
        
        self.density=0
        self.query_semantics=0
        self.concise=0
        
        self.interpretability_score=0
        self.llm = llm
        
        self._semantics_cache = semantics_cache if semantics_cache is not None else {}
        self._concise_cache = concise_cache if concise_cache is not None else {}
        
        self.prune = prune
        
    def calculate_scores(self):
        #todo change this into unique num
#         if t.size < 1: 
#             self.interpretability_score=0
#             return
        
#         self.density = self.calculate_density(t) 
        if not self.prune: self.density, total_size = self.estimate_density(self.original_table) 
        else:
            group_attrs = self.to_list(self.query.get('index')) + self.to_list(self.query.get('column'))
            self.density, total_size = 0, self.get_table_size(self.original_table, group_attrs)
        
        def to_list(val):
            if isinstance(val, list):
                return val
            elif val is None:
                return []
            else:
                return [val]
    
        def get_unique_vals_from_query_keys(query, unique_dict, keys):
            attrs = to_list(query.get(keys))
            return [unique_dict.get(attr, []) for attr in attrs]

        index_vals = get_unique_vals_from_query_keys(self.query, self.unique_dict, "index")
        column_vals = get_unique_vals_from_query_keys(self.query, self.unique_dict, "column")
        group_vals = index_vals + column_vals
        self.query_semantics = self.calculate_groupby_semantic(group_vals)

        self.concise = self.calculate_conciseness(total_size, tau_c=16, z=0.05, _lambda=0.5)
#         print(f"Density score: {self.density}")
#         print(f"Query Semantics score: {self.query_semantics}")
#         print(f"Conciseness score: {self.concise}")
        
        if not self.prune: self.interpretability_score = 1/3*(self.density+self.query_semantics+self.concise)
        else: self.interpretability_score = 1/2*(self.query_semantics+self.concise)
#         print(f"interpretability_score score: {self.interpretability_score}")

    # use materialization
    def calculate_density(self, t):
        nan_count = t.isna().sum().sum()
        total_num = t.size
        ratio = (nan_count/float(total_num))
        
        return 1.0 - ratio
    
    def get_table_size(self, df:pd.DataFrame, cols, treat_nan_as_value: bool = False)-> float:
        cardinalities = [df[col].nunique(dropna=not treat_nan_as_value) for col in cols]
        total_possible = np.prod(cardinalities)
        return total_possible
        
    def to_list(self, val):
        if isinstance(val, list):
            return val
        elif val is None:
            return []
        else:
            return [val]
        
    def estimate_density(self, df: pd.DataFrame, treat_nan_as_value: bool = False) -> tuple[float, Optional[int]]:
        """
        Estimate the density of a pivot table defined by group_attrs,
        without materializing the pivot.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            group_attrs (list of str): Columns to group by (row + column attrs).
            treat_nan_as_value (bool): Whether to treat NaNs as distinct values.

        Returns:
            float: Estimated density in [0, 1].
        """
        group_attrs = self.to_list(self.query.get('index')) + self.to_list(self.query.get('column'))
        if not group_attrs:
            return 0.0, None

        working_df = df[group_attrs].copy()

        if treat_nan_as_value:
            working_df = working_df.fillna("__NULL__")

        try:
            actual_groups = len(working_df.drop_duplicates())

            cardinalities = self.get_table_size(working_df, group_attrs, treat_nan_as_value)
            total_possible = np.prod(cardinalities)

            if total_possible == 0:
                return 0.0, None

            density = actual_groups / total_possible
            return min(density, 1.0), total_possible 

        except Exception as e:
            print(f"Error computing density: {e}")
            return 0.0, None
        
    def calculate_groupby_semantic(self, cols: List[List[Any]]) -> float:
        """
        Estimate how string-dominant each column is, and return the average match score.
        Each 'col' in 'cols' is a list of unique values (from one group-by attribute).
        """
        matches = []
        aggfuncs = self.query.get('aggfunc', [])
        val_names = self.query.get('value', [])

        # Ensure both are lists
        if not isinstance(aggfuncs, list):
            aggfuncs = [aggfuncs]
        if not isinstance(val_names, list):
            val_names = [val_names]

        try:
            cols_key = tuple(tuple(col) for col in cols)
        except Exception:
            cols_key = tuple(tuple([str(v) for v in col]) for col in cols)

        if not isinstance(aggfuncs, list):
            aggfuncs = [aggfuncs]
        if not isinstance(val_names, list):
            val_names = [val_names]

        key = (cols_key, tuple(sorted([str(a).upper() for a in aggfuncs])), tuple(sorted([str(v) for v in val_names])))
        if key in self._semantics_cache:
            return self._semantics_cache[key]

        for col_vals in cols:
            if len(col_vals) == 0:
                continue

            values = pd.Series(
                [val[-1] if isinstance(val, tuple) else val for val in col_vals],
                dtype=str
            )

            letters_count = values.str.count(r'\D')
            total_counts = values.str.len()

            total_letters = letters_count.sum()
            total_characters = total_counts.sum()

            string_dominant = total_letters / total_characters if total_characters > 0 else 0.0
            match = string_dominant

            # Use average rank weight across all value-aggfunc pairs
            aggfunc_rank_vals = []
            for val_name in val_names:
                aggfunc_rank = self.aggfunc_ranks.get(val_name, [])
                for aggfunc in aggfuncs:
                    aggfunc_upper = str(aggfunc).upper()
                    if aggfunc_upper in aggfunc_rank:
                        rank_index = aggfunc_rank.index(aggfunc_upper)
                        rank_val = RANK_MAP.get(rank_index, 0.0)
                    else:
                        rank_val = 0.0
                    aggfunc_rank_vals.append(rank_val)

            rank_score = sum(aggfunc_rank_vals) / len(aggfunc_rank_vals) if aggfunc_rank_vals else 0.0
            matches.append(match * rank_score)

        result = sum(matches) / len(matches) if matches else 0.0
        self._semantics_cache[key] = result
        return result

    def calculate_conciseness(self, t_size, tau_c=16, z=0.03, _lambda=0.5):
        if t_size in self._concise_cache:
            return self._concise_cache[t_size]

        if t_size is None:
            concise_score = 0.0
        elif t_size < tau_c:
            concise_score = 1 - z * t_size
        else:
            concise_score = (1 - z * tau_c) * np.exp(-_lambda*(t_size - tau_c))

        self._concise_cache[t_size] = concise_score
        return concise_score