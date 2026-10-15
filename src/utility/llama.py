import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
from collections import Counter
import os 
from src.utility.pivot_formatter import *
from src.config import *
import time
import json
from typing import List

REQUIRED_KEYS = {
    "correlation": {
        "instruction": {"groupA", "groupB", "column_attributes", "aggfunc", "value_attribute", "magnitude", "sign"},
        "prompt": {"instruction"},
    },
    "ratio": {
        "instruction": {"groupA", "groupB", "column_attributes", "aggfunc", "value_attribute", "magnitude"},
        "prompt": {"instruction"},
    },
    "outlier": {
        "instruction": {"groups", "aggfunc", "value_attribute"},
        "prompt": {"instruction"},
    },
    "interesting_columns": {
        "instruction": {},
        "prompt": {"instruction", "table"},
    },
    "interesting_aggfunc": {
        "instruction": {},
        "prompt": {"instruction", "table"},
    },
    "paraphrase": {
        "instruction": {},
        "prompt": {"instruction"},
    },
}  

class Prompt:
    def __init__(self, prompt_path=PROMPT_PATH, instruction_path=INSTRUCTION_PATH, 
                 pt:pd.DataFrame=None, value:str=None, aggfunc:str='mean', unique_dict:dict=None):
        self.prompt_path = prompt_path
        self.instruction_path = instruction_path
        if pt is not None: self.ptf = PivotTableFormatter(pt, value, aggfunc, unique_dict)
        
    def load_instruction(self, task_name: str, keywords: dict):
        file_path = os.path.join(self.instruction_path, f"{task_name}.txt")
        with open(file_path, "r") as f:
            instruction=f.read()
        
        required_keys = REQUIRED_KEYS[task_name]["instruction"]
        instruction = instruction.format_map(SafeDict(keywords, task_name=task_name, required_keys=required_keys))
        return instruction
    
    def load_prompt(self, task_name: str, keywords: dict):       
        file_name = task_name
        if not 'interesting' in file_name: file_name = 'likelihood'

        file_path = os.path.join(self.prompt_path, f"{file_name}.txt")
        with open(file_path, "r") as f:
            template=f.read()
        
        required_keys = REQUIRED_KEYS[task_name]["prompt"]
        template = template.format_map(SafeDict(keywords, task_name=task_name, required_keys=required_keys))
        return template
    
    def prepare_interesting_column(self, data):
         return {"table": self.dataframe_to_str(data)}

    def prepare_interesting_aggfunc(self, data, header):
        return {"table": self.series_to_str(data, header)}

    def prepare_correlation(self, groupA_index, groupB_index, magnitude, sign:str,isRow=True):
        vars_dict = self.ptf.get_correlation_prompt_vars(groupA_index, groupB_index, isRow)
        vars_dict['magnitude'] = magnitude
        vars_dict['sign'] = sign
        
        return vars_dict

    def prepare_correlation_clf(self, groupA_index, groupB_index, isRow=True):
        vars_dict = self.ptf.get_correlation_clf_prompt_vars(groupA_index, groupB_index, isRow)
        return vars_dict
    
    def prepare_ratio(self, groupA_index, groupB_index, magnitude,isRow=True):
        vars_dict = self.ptf.get_ratio_groups(groupA_index, groupB_index, isRow)
        vars_dict['magnitude'] = magnitude
        
        return vars_dict

    def prepare_ratio_clf(self, groupA_index, groupB_index, isRow=True):
        vars_dict = self.ptf.get_ratio_clf_prompt_vars(groupA_index, groupB_index, isRow)
        return vars_dict

    def prepare_outlier(self, groupA_index, groupB_index,isRow=True):
        vars_dict = self.ptf.get_outlier_groups(groupA_index, groupB_index, isRow)
        
        return vars_dict

    def prepare_outlier_clf(self, groupA_index, groupB_index, isRow=True):
        vars_dict = self.ptf.get_outlier_clf_prompt_vars(groupA_index, groupB_index, isRow)
        return vars_dict
    
    def dataframe_to_str(self, df):
        headers = list(df.columns.astype(str))
        header_row = "| " + " | ".join(headers) + " |"
        
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        data_rows = []
        for row in df.itertuples(index=False, name=None):
            data_rows.append("| " + " | ".join(map(str, row)) + " |")
        
        str_values = "\n".join([header_row, separator_row] + data_rows)
        
        return str_values
        
    def series_to_str(self, col:pd.Series, header_name:str):
        str_values = ' |\n| '.join(col.astype(str))
        str_values = '| ' + header_name + ' |\n| ' + '| --- |' + ' |\n| ' + str_values
        str_values = str_values + ' |\n' 
        return str_values


class Llama3:
    def __init__(self, prompt_path=PROMPT_PATH, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initializes the LLM withgiven model_path

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
        
        self.completion_keys = {"interesting_columns":"chosen_columns",
                        "interesting_aggfunc":"ranked_aggregation_functions",
                        "correlation": "likelihood",
                        "ratio":"likelihood",
                        "outlier":"likelihood",
                        }
        
    def get_response(
          self, conversations: str | List[str], message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
      ):
        prompts = [
        self.pipeline.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
        ]
        outputs = self.pipeline(
            prompts,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id = self.pipeline.tokenizer.eos_token_id
        )
        responses = [
        out[0]["generated_text"][len(prompt):] for out, prompt in zip(outputs, prompts)
        ]

        del prompts 
        del outputs

        return responses

    def chatbot(self, queries: str | List[str], system_instructions=""):
        is_batch = isinstance(queries, list)
        queries = [queries] if not is_batch else queries
    
        conversations = []
        for query in queries:
            conversation = [{"role": "system", "content": system_instructions},
                            {"role": "user", "content": query}]
            conversations.append(conversation)
        
        start_time = time.time()
        responses = self.get_response(conversations)
        end_time = time.time()

        elapsed_time = end_time - start_time
        self.log_response_time(elapsed_time)  # Log the timing
        
        return responses if is_batch else responses[0]
    
    def log_response_time(self, elapsed_time, log_file="llm_response_times.jsonl"):
        with open(log_file, "a") as f:
            f.write(json.dumps({"time": elapsed_time}) + "\n")
    
    def get_paraphrase_prompt(self, keywords: dict):
        task_name = "paraphrase"
        file_path = os.path.join(PROMPT_PATH, "paraphrase_instruction.txt")
        with open(file_path, "r") as f:
            prompt=f.read()

        required_keys = REQUIRED_KEYS[task_name]["prompt"]
        prompt = prompt.format_map(SafeDict(keywords, task_name=task_name, required_keys=required_keys))
        return prompt
    
    def paraphrase_instruction(self, instruction, num_paraphrase=4, include_original=True):
        paraphrases = []
        paraphrased_instruction = instruction

        while len(paraphrases) < num_paraphrase:
            prompt = self.get_paraphrase_prompt({"instruction":paraphrased_instruction})
            response = self.chatbot(prompt)
            paraphrased_instruction = self.find_value_in_response(response, "paraphrased_sentence")

            if paraphrased_instruction not in paraphrases:
                paraphrases.append(paraphrased_instruction)

        if include_original: paraphrases.append(instruction)
        
        return paraphrases
    
    def get_completion(self, instruction, pr:Prompt, task_name: str, do_paraphrase=False, keywords:dict=None):
        instructions = [instruction]
        if do_paraphrase:
            instructions = self.paraphrase_instruction(instructions[0])
        
        results=[]
        for idx, inst in enumerate(instructions):
            keywords['instruction']=inst
            prompt= pr.load_prompt(task_name, keywords)
            response=self.chatbot(prompt)
            values = self.find_value_in_response(response, self.completion_keys[task_name])
            results.append(values)
        
        results= majority_voting(results)
        
        return results
       
    def find_value_in_response(self, response, key):
        key_idx = response.find(f'"{key}"')

        if key_idx == -1:
            return None  

        start_idx = response.rfind('{', 0, key_idx)
        end_idx = response.find('}', key_idx)
    
        if start_idx == -1 or end_idx == -1:
            return None  

        json_str = response[start_idx:end_idx+1]

        try:
            data = json.loads(json_str)
            return data.get(key)
        except:
            return None
        return data[key]
    
class SafeDict(dict):
    class _SafeMissing:
        def __init__(self, key):
            self.key = key
        def __format__(self, spec):
            return f"{{{self.key}}}"

    def __init__(self, data, task_name=None, required_keys=None):
        super().__init__(data)
        self.task_name = task_name
        self.required_keys = required_keys or set()
        self._validate_required_keys()
        
    def _validate_required_keys(self):
        missing = self.required_keys - self.keys()
        if missing:
            raise ValueError(f"[{self.task_name}] Missing required keys: {missing}")

    def __missing__(self, key):
        return self._SafeMissing(key)
    
def majority_voting(data: list):
    flattened_data = []
    for sublist in data:
        for item in sublist:
            flattened_data.append(item)

    counter = Counter(flattened_data)

    threshold = len(data)//2

    majority_vote_items = [item for item, count in counter.items() if count >= threshold]

    return majority_vote_items

def dataframe_to_str_embedding(df):
    headers = [df.index.name] + list(df.columns.astype(str))
    header_row = "| " + " | ".join(headers) + " |"
    
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    data_rows = []
    for index, row in df.iterrows():
        row_values = [index] + list(row) 
        data_rows.append("| " + " | ".join(map(str, row_values)) + " |")
    
    str_values = "\n".join([header_row, separator_row] + data_rows)
    
    return str_values


def dataframe_to_str_embedding_no_special_tokens(df):
    headers = [df.index.name] + list(df.columns.astype(str))
    header_row = [" ".join(headers)]
    
    data_rows = []
    for index, row in df.iterrows():
        row_values = [index] + list(row) 
        data_rows.append(" ".join(map(str, row_values)))
    
    str_values = "\n".join(header_row + data_rows)
    
    return str_values

def dataframe_to_str_embedding_no_special_tokens_columnwise(pivot_table):
    text_result = []
    pivot_table=pivot_table.fillna('')

    text_result.append(pivot_table.index.name.lower())  
    text_result.extend(map(str, pivot_table.index))
    
    if pivot_table.columns.name:
        text_result.append(str(pivot_table.columns.name).lower()) 

    for col in pivot_table.columns.get_level_values(0):
        text_result.append(str(col).lower()) 
        text_result.extend(map(str, pivot_table[col].values))

    result_string = ' '.join(text_result)

    return result_string

def dataframe_to_str_embedding_no_special_tokens_descritions(pivot_table, comb):
    text_result = []
    pivot_table=pivot_table.fillna('')
    text_result.append(pivot_table.index.name.lower())  

    if comb['column']==[]:
        text_result.append("none") 
    else: text_result.append(comb['column'].lower()) 
    text_result.append(comb['value'].lower())
    text_result.append(comb['agg'].lower())

    if pivot_table.columns.name:
        text_result.append(str(pivot_table.columns.name).lower()) 

    text_result.append(pivot_table.index.name.lower())  
    text_result.extend(map(str, pivot_table.index[:10]))
        
    for col in pivot_table.columns.get_level_values(0):
        text_result.append(str(col).lower()) 

    result_string = ' '.join(text_result)
    print(f"Embedding input: {result_string}")
    return result_string

def dataframe_to_str_embedding_tokens_descritions(pivot_table, comb):
    text_result = []
    pivot_table=pivot_table.fillna('')
    
    index = pivot_table.index.name.lower()
    value = comb['value'].lower()
    aggfunc = comb['aggfunc'].lower()
    column=None
    
    cur_agg = aggfunc
    if aggfunc == 'min': cur_agg = 'minimum'
    elif aggfunc == 'max': cur_agg = 'maximum'
    elif aggfunc == 'mean': cur_agg = 'average'
        
    if comb['column']==[]:
        sentence = f"{cur_agg} {value} grouped by {index}."
    else:
        column = comb['column'].lower()
        sentence = f"{cur_agg} {value} grouped by {index} and {column}."
    
    text_result.append(sentence)
    text_result.append("[SEP]")
    
    values = []
    row_labels = list(map(str, pivot_table.index))
    col_labels = pivot_table.columns.get_level_values(0)
    
    for row_idx, row_label in enumerate(row_labels):
        if column is None:
            for col in col_labels:
                    cell = pivot_table.iloc[row_idx][col]
                    values.append(f"{cur_agg} {row_label} is {cell}")
        else:
            if row_idx < len(col_labels):
                cell = pivot_table.iloc[row_idx, row_idx]
                values.append(f"{cur_agg} {row_label} and {str(col_labels[row_idx])} is {cell}")

    value_results = ', '.join(values[:50])

    result_string = ' '.join(text_result)
    result_string += value_results

    return result_string

def dataframe_to_str_embedding_values_remove(pivot_table, comb):
    text_result = []
    pivot_table=pivot_table.fillna('')
    
    index = pivot_table.index.name.lower()
    value = comb['value'].lower()
    aggfunc = comb['agg'].lower()
    column=None
    if comb['column']==[]:
        sentence = f"The table is about {value} grouped by {index}"
    else:
        column = comb['column'].lower()
        sentence = f"The table is about {value} grouped by {index} and {column}"
    
    text_result.append(sentence)
    # text_result.append("[SEP]")
    # text_result.append("[CLS]")
    
    # if pivot_table.columns.name:
        # text_result.append(str(pivot_table.columns.name).lower()) 

    # text_result.append(pivot_table.index.name.lower())  
    row_labels = list(map(str, pivot_table.index))
    col_labels = pivot_table.columns.get_level_values(0)
    
    values = ["The values are grouped by"]
    for row_idx, row_label in enumerate(row_labels):
        if column is None:
            # for col in col_labels:
                # cell = pivot_table.iloc[row_idx][col]
            values.append(f"{row_label}")
        else:
            if row_idx < len(col_labels):
                values.append(f"{row_label} and {str(col_labels[row_idx])}")
        
    text_result.extend(values[:10])

    result_string = ' '.join(text_result)
    # print(f"Embedding input: {result_string}")
    return result_string
