'''
Implementation for insightfulness.
Section 3.1 in the paper.
'''
import pandas as pd
import numpy as np
from scipy.stats import skew
from copy import deepcopy
from scipy.spatial.distance import pdist
from numba import njit
from typing import Tuple, List
from src.utility.llama import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import GaussianRandomProjection
import random 
from scipy.stats import pearsonr
import math 

LIKELIHOOD_MAP={
    "Very Likely":0.2,
    "Likely":0.4,
    "Neutral":0.6,
    "Unlikely":0.8,
    "Very Unlikely":1.0
    }

        
LINEAR_SAMPLE_PERCENTAGE = 0.1 
QUADRATIC_SAMPLE_PERCENTAGE = 0.0001 
EPSILON = 0.2

from numba import njit
import numpy as np

@njit
def compute_high_ratio_pairs(arr, tau=0.8):
    n, d = arr.shape
    num_pairs = n * (n - 1) // 2
    scores = np.zeros(num_pairs)
    i_list = np.empty(num_pairs, dtype=np.int64)
    j_list = np.empty(num_pairs, dtype=np.int64)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            min_ab = np.inf
            min_ba = np.inf
            for k in range(d):
                a, b = arr[i, k], arr[j, k]

                if b != 0:
                    val_ab = a / b
                    if val_ab < min_ab:
                        min_ab = val_ab

                if a != 0:
                    val_ba = b / a
                    if val_ba < min_ba:
                        min_ba = val_ba

            score = max(min_ab, min_ba)
            scores[idx] = score
            i_list[idx] = i
            j_list[idx] = j
            idx += 1

    # Count number of scores > tau
    count = 0
    for i in range(num_pairs):
        if scores[i] > tau:
            count += 1

    # Allocate filtered output arrays
    indices = np.empty((count, 2), dtype=np.int64)  # for (i, j)
    scores_ = np.empty(count, dtype=np.float64)
    idx_out = 0
    for i in range(num_pairs):
        if scores[i] > tau:
            indices[idx_out, 0] = i_list[i]
            indices[idx_out, 1] = j_list[i]
            scores_[idx_out] = scores[i]
            idx_out += 1

    return indices, scores_

def convert_trend_vars_to_feature(vars: dict, index_attr: str, insight_type="corr", bucket="unknown") -> dict:
    return {
        "index_attr": index_attr,
        "index_values": '|'.join(sorted([str(vars["groupA"]), str(vars["groupB"])])),
        "column_attr": vars["column_attributes"],
        "value_attr": vars["value_attribute"],
        "aggfunc": vars["aggfunc"],
        "insight_type": insight_type,
        "bucket": bucket
    }


def bucketize_correlation(value: float) -> str:
    if value >= 0.9:
        return 'very high'
    elif value >= 0.7:
        return 'high'
    elif value >= 0.5:
        return 'neutral'
    elif value >= 0.3:
        return 'low'
    else:
        return 'very low'
    
def bucketize_ratio(value: float) -> str:
    if value <= 1.5:
        return 'similar magnitude'
    elif value <= 3.0:
        return 'moderate difference'
    elif value <= 10.0:
        return 'large difference'
    elif value <= 30.0:
        return 'very large difference'
    else:
        return 'extreme difference'

class NaiveInsightfulness:
    def __init__(self, pt, d_size=None, t_size=100, original_table=None, query=None, llm: Llama3=None, pr:Prompt=None,
                 tau_cor:float=0.7, tau_rat:float=5.0, clf_trd=None, clf_out=None, gamma=None):
        # df : NxM
        self.d_size = d_size
        self.t_size = t_size
        
        self.query = query
        self.original_table = original_table
        
        if query: self.aggfunc = query['aggfunc']
        
        self.informativeness_score=0
        
        self.interesting_attributes = None
        self.correlation_score=0
        self.ratio_score=0
        self.trend_score=0
        
        self.surprise_score=0
        
        self.insightfulness_score=0
        
        self.llm = llm
        self.pr = pr
        
        self.tau_cor = tau_cor
        self.tau_rat = tau_rat
        
        self.clf_trd = clf_trd
        self.clf_out = clf_out
        
        self.gamma = gamma
        
    def calculate_scores(self, t):
        ##########################
        # interesting attributes #
        ##########################
        # self.interesting_attributes = self.get_interesting_attributes(t)
        
        #######################
        # Calculate Deviation #
        #######################
        self.informativeness_score=self.calculate_inf(t.fillna(0))

        #########################
        # Calculate Correlation #
        #########################
        self.correlation_score = self.calculate_correlation(t.fillna(0), tau=self.tau_cor)
        self.ratio_score = self.calculate_ratio(t.fillna(0))
        self.trend_score = max(self.correlation_score, self.ratio_score)
                
        ############################
        # Calculate Surprisingness #
        ############################
        self.surprise_score=self.calculate_surprising(t)

        self.insightfulness_score = max(self.informativeness_score, self.trend_score,  self.surprise_score)

    def get_likelihood_clf(self, task_name: str = 'correlation', vars: dict = None) -> float:
        if vars is None:
            print("Warning: 'vars' is None, returning 0.")
            return 0

        x_df = pd.DataFrame([vars])

        if task_name != 'outlier':
            if self.clf_trd:
                try:
                    y_pred = self.clf_trd.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        else:
            if self.clf_out:
                try:
                    y_pred = self.clf_out.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        
    def get_likelihood(self, task_name: str = 'correlation', vars: dict = None) -> float:
        instruction = self.pr.load_instruction(task_name, vars)
        prompt = self.pr.load_prompt(task_name, {"instruction": instruction})

        response = self.llm.chatbot(prompt)
        result= self.llm.find_value_in_response(response, self.llm.completion_keys[task_name])
        final_result = LIKELIHOOD_MAP[result]
                
        return final_result
        
    def calculate_inf(self, t: pd.DataFrame) -> float:
        n, m = t.shape
        
        row_pairwise_distances, col_pairwise_distances = None, None
        
        def compute_pairwise_euclidean(arr, norm_size, gamma):
            pairwise_distance = pdist(arr, metric='euclidean')
            pairwise_distance /= norm_size
            pairwise_distance /= gamma
            
            return pairwise_distance
        
        if n == 1: row_informativeness_score = 0.0
        else:
            data_array = t.values
            row_pairwise_distances = compute_pairwise_euclidean(data_array, m, self.gamma)
            row_informativeness_score = np.sum(row_pairwise_distances)/ (n * (n - 1) / 2)
            
        if m == 1: col_informativeness_score = 0.0
        else:
            if n == 1:
                data_array = t.T[1:].to_numpy().reshape(-1, 1)
            else:
                data_array = t.T.values
            col_pairwise_distances = compute_pairwise_euclidean(data_array, n, self.gamma)
            col_informativeness_score = np.sum(col_pairwise_distances)/ (m * (m - 1) / 2)

        informative_score = max(row_informativeness_score, col_informativeness_score)
        
        return informative_score
    
    def calculate_correlation(self, t, tau=0.8):
        n, m = t.shape
        correlation_score = None
        row_corr_score, col_corr_score = None, None
        row_pairwise_corr, col_pairwise_corr = None, None
        data_array = t.values
        
        def compute_pairwise_correlation(arr, IsRow=True):
            corr = np.corrcoef(arr, rowvar=IsRow)
            corr_vals = corr[np.triu_indices(corr.shape[0], k=1)]
            corr_vals = np.abs(corr_vals)
    
            corr_vals = np.nan_to_num(corr_vals, nan=0)

            return corr_vals
        
        # def compute_high_corr_pairs(arr, IsRow=True)->List:
        #     if not IsRow: arr = arr.T
        #     corr = np.corrcoef(arr)
        #     corr = np.nan_to_num(corr, nan=0)

        #     i_indices, j_indices = np.triu_indices(corr.shape[0], k=1)

        #     high_corr_pairs_vals = [
        #         (i, j, corr[i,j], np.abs(corr[i,j])) for i, j in zip(i_indices, j_indices) if np.abs(corr[i, j]) > self.tau_cor
        #     ]
        #     return high_corr_pairs_vals
        
        def compute_high_corr_pairs(arr, tau_cor, IsRow=True):
            if not IsRow:
                arr = arr.T

            corr = np.corrcoef(arr)
            corr = np.nan_to_num(corr, nan=0)

            i_indices, j_indices = np.triu_indices(corr.shape[0], k=1)
            corr_vals = corr[i_indices, j_indices]
            abs_corr_vals = np.abs(corr_vals)

            mask = abs_corr_vals > tau_cor

            return list(zip(
                i_indices[mask],
                j_indices[mask],
                corr_vals[mask],
                abs_corr_vals[mask]
            ))
    
        def get_correlation_likelihood(i, j, val, IsRow):
            if self.clf_trd:
                cor_vars=self.pr.prepare_correlation_clf(i,j,isRow=IsRow)
                cor_vars['bucket'] = bucketize_correlation(abs(val))
                result = self.get_likelihood_clf('correlation',cor_vars)
            else:
                sign = "positive" if val >= 0 else "negative"
                cor_vars=self.pr.prepare_correlation(i,j,val,sign,isRow=IsRow)
                result = self.get_likelihood('correlation',cor_vars)
            return result

        if n == 1: # Case with only one row
            row_corr_score = [0]
        elif np.all(np.std(data_array, axis=1) == 0):
            row_corr_score = [0]
        else:
            likelihoods, pairwise_corr = None, None
            high_corr_pairs_vals = compute_high_corr_pairs(data_array, tau_cor=tau, IsRow=True)
            if not high_corr_pairs_vals:
                row_pairwise_corr = 0.0
            else:
                likelihoods = [get_correlation_likelihood(a, b, c, True) for a, b, c, d in high_corr_pairs_vals]
                pairwise_corr = [d for a, b, c, d in high_corr_pairs_vals]
                row_pairwise_corr = [l * r for l, r in zip(likelihoods, pairwise_corr)]
            row_corr_score = np.sum(row_pairwise_corr) / (n * (n - 1) / 2)
            
        if m == 1:  # Case with only one column
            col_corr_score = [0]
        elif np.all(np.std(data_array, axis=0) == 0):
            col_corr_score = [0]
        else:
            high_corr_pairs_vals = compute_high_corr_pairs(data_array, tau_cor=tau, IsRow=False)
            if not high_corr_pairs_vals:
                col_pairwise_corr = 0.0
            else:
                likelihoods = [get_correlation_likelihood(a, b, c, False) for a, b, c, d in high_corr_pairs_vals]
                pairwise_corr = [d for a, b, c, d in high_corr_pairs_vals]
                col_pairwise_corr = [l * r for l, r in zip(likelihoods, pairwise_corr)]
            col_corr_score = np.sum(col_pairwise_corr) / ((m * (m - 1) / 2))

        correlation_score = max(row_corr_score, col_corr_score)

        return correlation_score
                
    def calculate_ratio(self, t, tau=2.0):
        n, m = t.shape
        
        row_rat_score, col_rat_score = None, None
        row_mins, col_mins = None, None 
    
        def get_inverese_ratio(arr):
            with np.errstate(divide='ignore', invalid='ignore'):
                inf_mask = arr != np.inf
                null_mask = arr != 0.0
                result = np.zeros_like(arr, dtype=np.float64)
                result[inf_mask & null_mask] = 1 - 1 / arr[inf_mask & null_mask]
                
                result = np.sum(result)
            return result
        
        def get_ratio_likelihood(i, j, val, isRow):
            if self.clf_trd:
                rat_vars=self.pr.prepare_ratio_clf(i,j,isRow=isRow)
                rat_vars['bucket'] = bucketize_ratio(abs(val))
                result = self.get_likelihood_clf('ratio',rat_vars)
            else:
                rat_vars=self.pr.prepare_ratio(i,j,val,isRow)
                result = self.get_likelihood('ratio', rat_vars)
            return result
        
        if n == 1: 
            row_rat_score = [0]
        else:
            data_array = t.values
            high_rat_pairs, high_rat_vals = compute_high_ratio_pairs(data_array, tau=self.tau_rat)

            if len(high_rat_pairs) == 0:
                row_rat_score = [0]
            else:
                high_rat_pairs = np.atleast_2d(high_rat_pairs)
                likelihoods = np.array([get_ratio_likelihood(a, b, c, isRow=True) for (a, b), c in zip(high_rat_pairs, high_rat_vals)])
                row_mins = np.array([l * r for l, r in zip(likelihoods, high_rat_vals)])
                
                row_rat_score = get_inverese_ratio(row_mins)/(n * (n - 1) / 2)
            
        if m == 1: 
            col_rat_score = [0]
        else:
            data_array = t.T.values
            high_rat_pairs, high_rat_vals = compute_high_ratio_pairs(data_array, tau=self.tau_rat)

            if len(high_rat_pairs) == 0:
                col_rat_score = [0]
            else:
                high_rat_pairs = np.atleast_2d(high_rat_pairs)
                likelihoods = np.array([get_ratio_likelihood(a, b, c, isRow=False) for (a, b), c in zip(high_rat_pairs, high_rat_vals)])
                col_mins = np.array([l * r for l, r in zip(likelihoods, high_rat_vals)])
                
                col_rat_score = get_inverese_ratio(col_mins)/(m * (m - 1) / 2)
            
        ratio_score = max(row_rat_score, col_rat_score)
        
        return ratio_score

    def calculate_surprising(self, t, threshold=4.0):
        n, m = t.shape
        
        row_sur_score, col_sur_score = None, None
        
        def compute_outlier(arr, threshold):
            mean = np.nanmean(arr, axis=1, keepdims=True) 
            std = np.nanstd(arr, axis=1, keepdims=True) 
            deviation = np.abs(arr - mean)
            
            mask = np.zeros_like(arr, dtype=bool)
            valid_std = std != 0  
            valid_std = np.broadcast_to(valid_std, arr.shape)
            
            deviation_mask = deviation >= std * threshold
            mask = deviation_mask & valid_std            
            
            return mask

        def get_outlier_likelihood(i, j, isRow):
            if self.clf_trd:
                outlier_vars=self.pr.prepare_outlier_clf(i,j,isRow=isRow)
                result = self.get_likelihood_clf('outlier',outlier_vars)
            else:
                outlier_vars=self.pr.prepare_outlier(i,j, isRow)
                result = self.get_likelihood('outlier', outlier_vars)
            return result
        
        if m == 1: 
            row_sur_score = 0.0
        else:
            data_array = t.values
            row_outlier_mask = compute_outlier(data_array, threshold)
            row_scores = []
            for i in range(n):
                outlier_cols = np.where(row_outlier_mask[i])[0]
                if len(outlier_cols) > 0:
                    likelihood_sum = 0.0
                    for j in outlier_cols:
                        likelihood_sum += get_outlier_likelihood(i, j, True)
                    score = 1.0 - (likelihood_sum / (len(outlier_cols) + 1))
                else:
                    score = 0.0
                row_scores.append(score)

            row_sur_score = sum(row_scores) / n
            
        if n == 1: 
            col_sur_score = 0.0
        else:
            data_array = t.T.values
            col_outlier_mask = compute_outlier(data_array, threshold) 
            col_scores = []
            for j in range(m):
                outlier_rows = np.where(col_outlier_mask[j])[0]
                if len(outlier_rows) > 0:
                    likelihood_sum = 0.0
                    for i in outlier_rows:
                        likelihood_sum += get_outlier_likelihood(j, i, isRow=False)
                    score = 1.0 - (likelihood_sum / (len(outlier_rows) + 1))
                else:
                    score = 0.0
                col_scores.append(score)

            col_sur_score = sum(col_scores) / m
                    
        surprise_score = max(row_sur_score, col_sur_score)
        
        return surprise_score


class ImprovedInsightfulness:
    def __init__(self, pt, d_size=None, t_size=100, original_table=None, query=None, llm: Llama3=None, pr:Prompt=None,
                 tau_cor:float=0.7, tau_rat:float=5.0, clf_trd=None, clf_out=None, gamma=None):
        # df : NxM
        self.d_size = d_size
        self.t_size = t_size
        
        self.query = query
        self.original_table = original_table
        
        if query: self.aggfunc = query['aggfunc']

        self.informativeness_score=0
        
        self.interesting_attributes = None
        self.correlation_score=0
        self.ratio_score=0
        self.trend_score=0
        
        self.surprise_score=0
        
        self.insightfulness_score=0
        
        self.llm = llm
        self.pr = pr
        
        self.tau_cor = tau_cor
        self.tau_rat = tau_rat
        
        self.clf_trd = clf_trd
        self.clf_out = clf_out
        
        self.gamma = gamma
        
    def calculate_scores(self, t):
        ##########################
        # interesting attributes #
        ##########################
        # self.interesting_attributes = self.get_interesting_attributes(t)
        
        n, m = t.shape
        
        # --- 1. DETERMINE ABSOLUTE SAMPLE SIZES ---

        N_max = max(n, m)
        num_possible_pairs = N_max * (N_max - 1) / 2
        
        k_pairs = int(math.ceil(num_possible_pairs * QUADRATIC_SAMPLE_PERCENTAGE))
        k_pairs = max(5000, k_pairs) 
        
        num_possible_samples = max(n, m) 
        
        k_samples = int(math.ceil(num_possible_samples * LINEAR_SAMPLE_PERCENTAGE))
        k_samples = max(500, k_samples)
        
        #######################
        # Calculate Deviation #
        #######################
        self.informativeness_score=self.calculate_inf_approx_rp(t.fillna(0), epsilon=EPSILON)

        #########################
        # Calculate Correlation #
        #########################
        self.correlation_score = self.calculate_correlation_approx(t.fillna(0), tau=self.tau_cor, k_pairs=k_pairs)
        self.ratio_score = self.calculate_ratio_approx(t.fillna(0), k_pairs=k_pairs)
        self.trend_score = max(self.correlation_score, self.ratio_score)
                
        ############################
        # Calculate Surprisingness #
        ############################
        self.surprise_score=self.calculate_surprising_approx(t, k_samples=k_samples)

        self.insightfulness_score = max(self.informativeness_score, self.trend_score,  self.surprise_score)

    def get_likelihood_clf(self, task_name: str = 'correlation', vars: dict = None) -> float:
        if vars is None:
            print("Warning: 'vars' is None, returning 0.")
            return 0

        x_df = pd.DataFrame([vars])

        if task_name != 'outlier':
            if self.clf_trd:
                try:
                    y_pred = self.clf_trd.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        else:
            if self.clf_out:
                try:
                    y_pred = self.clf_out.predict(x_df)[0]
                    return LIKELIHOOD_MAP.get(y_pred, 0)
                except Exception as e:
                    return 0
            else:
                return 0

        
    def get_likelihood(self, task_name: str = 'correlation', vars: dict = None) -> float:
        instruction = self.pr.load_instruction(task_name, vars)
        prompt = self.pr.load_prompt(task_name, {"instruction": instruction})

        response = self.llm.chatbot(prompt)
        result= self.llm.find_value_in_response(response, self.llm.completion_keys[task_name])
        final_result = LIKELIHOOD_MAP[result]
                
        return final_result
        
    def calculate_inf_approx_rp(self, t: pd.DataFrame, epsilon: float = 0.1) -> float:
        n, m = t.shape
        
        # JL-lemma target dimension
        num_points = max(n, m)
        if num_points <= 1: 
            return 0.0
        
        k = int(np.ceil(np.log(num_points) / (epsilon**2)))
        
        k = min(k, m - 1) # For row projection
        k_prime = min(k, n - 1) # For column projection
        
        if k < 1: k = 1
        if k_prime < 1: k_prime = 1
        
        def compute_approximate_score_rp(data_array, D_orig, N_size, k_proj):
            if N_size <= 1:
                return 0.0
            
            # Only project if the original dimension D_orig is larger than k_proj
            if D_orig > k_proj:
                # Create a random projector (the sketching component)
                projector = GaussianRandomProjection(n_components=k_proj)
                # Project the data (n x D_orig) -> (n x k_proj)
                data_array_sketch = projector.fit_transform(data_array)
            else:
                data_array_sketch = data_array # No projection needed/possible

            # Now compute pdist on the sketched data
            pairwise_distance = pdist(data_array_sketch, metric='euclidean')
            
            # Apply normalization based on *original* dimension D_orig
            pairwise_distance /= D_orig 
            pairwise_distance /= self.gamma
            
            num_possible_pairs = N_size * (N_size - 1) / 2
            return np.sum(pairwise_distance) / num_possible_pairs

        # --- Row Informativeness ---
        data_array_rows = t.values
        row_informativeness_score = compute_approximate_score_rp(
            data_array_rows, m, n, k
        )

        # --- Column Informativeness ---
        data_array_cols = t.T.values
        col_informativeness_score = compute_approximate_score_rp(
            data_array_cols, n, m, k_prime
        )

        informative_score = max(row_informativeness_score, col_informativeness_score)
        
        return informative_score
        
    def calculate_correlation_approx(self, t: pd.DataFrame, tau: float = 0.8, k_pairs: int = 5000, approximate: bool = True):
        n, m = t.shape
        data_array = t.values

        def bucketize_correlation(abs_val):
            return int(abs_val * 10) # Example: maps 0.8 to 8

        def get_correlation_likelihood(i, j, val, IsRow):
            if self.clf_trd:
                cor_vars = self.pr.prepare_correlation_clf(i, j, isRow=IsRow)
                cor_vars['bucket'] = bucketize_correlation(abs(val))
                result = self.get_likelihood_clf('correlation', cor_vars)
            else:
                sign = "positive" if val >= 0 else "negative"
                cor_vars = self.pr.prepare_correlation(i, j, val, sign, isRow=IsRow)
                result = self.get_likelihood('correlation', cor_vars)
            return result
        
        def compute_approx_corr_score(arr, tau_cor, IsRow, k):
            N_size = arr.shape[0] # Number of vectors (n or m)
            
            if N_size <= 1:
                return 0.0
                
            num_possible_pairs = N_size * (N_size - 1) // 2
            
            if not approximate or k >= num_possible_pairs:
                # exact
                corr = np.corrcoef(arr)
                corr = np.nan_to_num(corr, nan=0)

                i_indices, j_indices = np.triu_indices(corr.shape[0], k=1)
                corr_vals = corr[i_indices, j_indices]
                abs_corr_vals = np.abs(corr_vals)

                mask = abs_corr_vals > tau_cor
                
                weighted_sum = 0.0
                for i, j, val in zip(i_indices[mask], j_indices[mask], corr_vals[mask]):
                    likelihood = get_correlation_likelihood(i, j, val, IsRow)
                    weighted_sum += likelihood * np.abs(val)
                    
                return weighted_sum / num_possible_pairs

            # approximate: score k random pairs
            indices = list(range(N_size))
            all_pairs = [(i, j) for i in range(N_size) for j in range(i + 1, N_size)]
            
            if k < len(all_pairs):
                sampled_pairs = random.sample(all_pairs, k)
            else:
                sampled_pairs = all_pairs # Fallback, though k check should prevent this

            sampled_scores = []
            for i, j in sampled_pairs:
                val, _ = pearsonr(arr[i], arr[j])
                
                if np.isnan(val):
                    val = 0.0
                
                abs_val = np.abs(val)
                
                if abs_val > tau_cor:
                    likelihood = get_correlation_likelihood(i, j, val, IsRow)
                    sampled_scores.append(likelihood * abs_val)
            
            # unbiased estimate of the exact score: sum / k
            if not sampled_scores:
                return 0.0
                
            return np.sum(sampled_scores) / k
            
        row_corr_score = compute_approx_corr_score(data_array, tau, IsRow=True, k=k_pairs)

        data_array_cols = data_array.T
        col_corr_score = compute_approx_corr_score(data_array_cols, tau, IsRow=False, k=k_pairs)

        correlation_score = max(row_corr_score, col_corr_score)

        return correlation_score
                    
    def calculate_ratio_approx(self, t: pd.DataFrame, k_pairs: int = 5000, approximate: bool = True):
        n, m = t.shape
        
        def bucketize_ratio(abs_val):
            return int(min(abs_val, 10))
        
        def get_ratio_likelihood(i, j, val, isRow):
            return 1.0
            
        def get_inverese_ratio(arr):
            with np.errstate(divide='ignore', invalid='ignore'):
                inf_mask = arr != np.inf
                null_mask = arr != 0.0
                result = np.zeros_like(arr, dtype=np.float64)
                result[inf_mask & null_mask] = 1 - 1 / arr[inf_mask & null_mask]
                
                return np.sum(result)

        def compute_high_ratio_pairs_for_pair(vector1, vector2, tau):
            # max element-wise ratio, if above tau
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.maximum(vector1 / vector2, vector2 / vector1)
            
            max_ratio = np.nanmax(ratios) if not np.all(np.isnan(ratios)) else 0.0
            
            if max_ratio > tau:
                return max_ratio
            return None

        def compute_approx_ratio_score(arr, tau_rat, IsRow, k):
            N_size = arr.shape[0] # Number of vectors (n or m)
            
            if N_size <= 1:
                return 0.0
                
            num_possible_pairs = N_size * (N_size - 1) // 2
            
            if not approximate or k >= num_possible_pairs:
                # exact
                high_rat_pairs, high_rat_vals = compute_high_ratio_pairs(arr, tau=tau_rat)
                if len(high_rat_pairs) == 0:
                    return 0.0
                else:
                    high_rat_pairs = np.atleast_2d(high_rat_pairs)
                    likelihoods = np.array([get_ratio_likelihood(a, b, c, IsRow) for (a, b), c in zip(high_rat_pairs, high_rat_vals)])
                    mins = np.array([l * r for l, r in zip(likelihoods, high_rat_vals)])
                    return get_inverese_ratio(mins) / num_possible_pairs


            # approximate: score k random pairs
            indices = list(range(N_size))
            all_pairs = [(i, j) for i in range(N_size) for j in range(i + 1, N_size)]
            
            if k < len(all_pairs):
                sampled_pairs = random.sample(all_pairs, k)
            else:
                sampled_pairs = all_pairs 

            weighted_ratios_list = []
            for i, j in sampled_pairs:
                vec_i, vec_j = arr[i], arr[j]
                
                ratio_val = compute_high_ratio_pairs_for_pair(vec_i, vec_j, tau_rat)
                
                if ratio_val is not None:
                    likelihood = get_ratio_likelihood(i, j, ratio_val, IsRow)
                    weighted_ratios_list.append(likelihood * ratio_val)
            
            if not weighted_ratios_list:
                return 0.0
                
            sum_of_inverse_ratios = get_inverese_ratio(np.array(weighted_ratios_list))
            # unbiased estimate of the exact score: sum / k
            return sum_of_inverse_ratios / k

        data_array_rows = t.values
        row_rat_score = compute_approx_ratio_score(data_array_rows, self.tau_rat, IsRow=True, k=k_pairs)

        data_array_cols = t.T.values
        col_rat_score = compute_approx_ratio_score(data_array_cols, self.tau_rat, IsRow=False, k=k_pairs)

        ratio_score = max(row_rat_score, col_rat_score)
        
        return ratio_score

    def calculate_surprising_approx(self, t: pd.DataFrame, threshold: float = 4.0, k_samples: int = 1000, approximate: bool = True):
        n, m = t.shape
        
        def compute_outlier(arr, threshold):
            mean = np.nanmean(arr, axis=1, keepdims=True) 
            std = np.nanstd(arr, axis=1, keepdims=True) 
            deviation = np.abs(arr - mean)
            
            mask = np.zeros_like(arr, dtype=bool)
            valid_std = std != 0  
            valid_std = np.broadcast_to(valid_std, arr.shape)
            
            deviation_mask = deviation >= std * threshold
            mask = deviation_mask & valid_std            
            
            return mask

        def get_outlier_likelihood(i, j, isRow):
            if self.clf_trd:
                outlier_vars=self.pr.prepare_outlier_clf(i,j,isRow=isRow)
                result = self.get_likelihood_clf('outlier',outlier_vars)
            else:
                outlier_vars=self.pr.prepare_outlier(i,j, isRow)
                result = self.get_likelihood('outlier', outlier_vars)
            return result
        
        # --- Row Surprise Score Calculation ---
        if m == 1: 
            row_sur_score = 0.0
        else:
            data_array = t.values
            row_outlier_mask = compute_outlier(data_array, threshold)
            row_scores = []
            
            # Determine which rows to process
            rows_to_check = list(range(n))
            if approximate and n > k_samples:
                rows_to_check = random.sample(rows_to_check, k_samples)

            for i in rows_to_check:
                outlier_cols = np.where(row_outlier_mask[i])[0]
                score = 0.0
                if len(outlier_cols) > 0:
                    likelihood_sum = 0.0
                    for j in outlier_cols:
                        likelihood_sum += get_outlier_likelihood(i, j, True)
                    score = 1.0 - (likelihood_sum / (len(outlier_cols) + 1))
                
                row_scores.append(score)

            if not row_scores:
                row_sur_score = 0.0
            else:
                # The score is always averaged by the number of samples checked (k_samples or n)
                row_sur_score = sum(row_scores) / len(rows_to_check)
                
        # --- Column Surprise Score Calculation ---
        if n == 1: 
            col_sur_score = 0.0
        else:
            data_array = t.T.values
            col_outlier_mask = compute_outlier(data_array, threshold)
            col_scores = []
            
            # Determine which columns to process
            cols_to_check = list(range(m))
            if approximate and m > k_samples:
                cols_to_check = random.sample(cols_to_check, k_samples)

            for j in cols_to_check: # Iterate through the transposed rows (which are the original columns)
                outlier_rows = np.where(col_outlier_mask[j])[0]
                score = 0.0
                if len(outlier_rows) > 0:
                    likelihood_sum = 0.0
                    for i in outlier_rows:
                        likelihood_sum += get_outlier_likelihood(j, i, isRow=False)
                    score = 1.0 - (likelihood_sum / (len(outlier_rows) + 1))
                
                col_scores.append(score)

            if not col_scores:
                col_sur_score = 0.0
            else:
                col_sur_score = sum(col_scores) / len(cols_to_check)
                        
        surprise_score = max(row_sur_score, col_sur_score)
        
        return surprise_score