"""LLM-proxy training data (Section 5.1.2): Llama-3 likelihood labels for sampled correlation, ratio and outlier configs.

    python -m src.prompt_learning.generate_dataset --dataset Video --max_columns 90 --out_dir result/proxy/data

Writes <out_dir>/<dataset>_<kind>.jsonl, one config per line; label is null if Llama gave none. Reruns resume.
Llama decodes greedily (do_sample=False); --sample samples them instead.
"""
import argparse
import json
import os
import random

import pandas as pd

from src.preprocessing import Preprocess
from src.utility.utility import retrieve_unique_values_from_df

AGGFUNCS = ['mean', 'count', 'min', 'max', 'sum']
CORR_BUCKETS = ['very high', 'high', 'neutral', 'low', 'very low']
RAT_BUCKETS = ['similar_magnitude', 'moderate difference', 'large difference', 'very large difference', 'extreme difference']
BUCKETS = {'correlation': [('corr', b) for b in CORR_BUCKETS], 'ratio': [('rat', b) for b in RAT_BUCKETS]}
KINDS = ['correlation', 'ratio', 'outlier']

TREND_KEYS = ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket']
OUTLIER_KEYS = ['index_attr', 'index_value', 'column_attr', 'column_value', 'value_attr', 'aggfunc']

PROMPT = """# Task Description: {instruction} Choose the appropriate likelihood from following scale. Return the result as JSON in the following format: {{"likelihood": "<one likelihood from the scale>"}}. Please return only the JSON output. Do not include explanations, code, or the full table.
    ## Input:
    **Likelihood Scale:**
    Very Likely
    Likely
    Neutral
    Unlikely
    Very Unlikely

    Return the result as JSON in the following format: {{"likelihood": "<one likelihood from the scale>"}}.

    ## Output:
    """


def load_table(filepath: str, filename: str, max_columns: int = None) -> pd.DataFrame:
    """The table as src/models/greedy_basic.py loads it (before its row cut, like its unique values)."""
    df = pd.read_csv(os.path.join(filepath, f"{filename}.csv"), index_col=False).dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].fillna(0)
    if max_columns is not None and max_columns < df.shape[1]:
        df = df.iloc[:, :max_columns]
    return Preprocess(df).do_preprocess()


def config_space(df: pd.DataFrame, interesting_attrs: list, kind: str):
    """(heads, tails): every config is a head (attributes and their values) combined with a tail."""
    unique_vals = retrieve_unique_values_from_df(df, interesting_attrs)
    group_cols = [col for col, vals in unique_vals.items() if 2 <= len(vals) <= 100]
    value_cols = df.select_dtypes(include='number').columns.tolist()

    heads = []
    for group_col in group_cols:
        vals = unique_vals[group_col]
        for col in group_cols:
            if col == group_col:
                continue
            if kind != 'outlier':
                heads += [(group_col, sorted([vals[i], vals[j]]), col)
                          for i in range(len(vals)) for j in range(i + 1, len(vals))]
            else:
                heads += [(group_col, idx_val, col, col_val) for idx_val in vals for col_val in unique_vals[col]]

    tails = [(v, a) for v in value_cols for a in AGGFUNCS]
    if kind != 'outlier':
        tails = [(v, a, t, b) for v, a in tails for t, b in BUCKETS[kind]]
    return heads, tails


def make_config(head, tail, kind: str) -> dict:
    if kind != 'outlier':
        (index_attr, index_values, column_attr), (value_attr, aggfunc, insight_type, bucket) = head, tail
        return {'index_attr': index_attr, 'index_values': index_values, 'column_attr': column_attr,
                'value_attr': value_attr, 'aggfunc': aggfunc, 'insight_type': insight_type, 'bucket': bucket}
    (index_attr, index_value, column_attr, column_value), (value_attr, aggfunc) = head, tail
    return {'index_attr': index_attr, 'index_value': index_value, 'column_attr': column_attr,
            'column_value': column_value, 'value_attr': value_attr, 'aggfunc': aggfunc}


def sample_configs(heads, tails, kind: str, n: int, seed: int) -> list:
    """n distinct configs, uniformly from the len(heads) x len(tails) space."""
    total = len(heads) * len(tails)
    picks = random.Random(seed).sample(range(total), min(n, total))
    return [make_config(heads[p // len(tails)], tails[p % len(tails)], kind) for p in picks]


def config_key(config: dict, kind: str) -> str:
    return json.dumps({k: config[k] for k in (OUTLIER_KEYS if kind == 'outlier' else TREND_KEYS)}, sort_keys=True)


def build_prompt(config: dict, kind: str) -> str:
    if kind == 'outlier':
        instruction = (
            f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_value']}' and "
            f"'{config['column_attr']} '{config['column_value']}' shows an outlier"
        )
    else:
        what = 'correlation' if config['insight_type'] == 'corr' else 'ratio difference'
        instruction = (
            f"The {config['aggfunc']} values of '{config['value_attr']}' for {config['index_attr']} '{config['index_values'][0]}' and "
            f"'{config['index_values'][1]}' show a {config['bucket']} {what} "
            f"when grouped by '{config['column_attr']}'."
        )
    return PROMPT.format(instruction=instruction)


def ask(llm, prompts: list, greedy: bool = True) -> list:
    """Llama's responses: greedy decoding with Llama3's chat template, or Llama3.chatbot's sampling."""
    if not greedy:
        return llm.chatbot(prompts)
    texts = [llm.pipeline.tokenizer.apply_chat_template(
                 [{"role": "system", "content": ""}, {"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
    outputs = llm.pipeline(texts, max_new_tokens=4096, do_sample=False, temperature=None, top_p=None,
                           pad_token_id=llm.pipeline.tokenizer.eos_token_id)
    return [out[0]["generated_text"][len(t):] for out, t in zip(outputs, texts)]


def label_configs(configs: list, kind: str, llm, out_path: str, batch_size: int = 64, greedy: bool = True):
    """Ask Llama about every config not yet in out_path, appending each batch as it finishes."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {config_key(json.loads(line), kind) for line in f if line.strip()}
    todo = [c for c in configs if config_key(c, kind) not in done]
    print(f"[{kind}] {len(configs)} configs, {len(done)} already labeled, {len(todo)} to ask -> {out_path}", flush=True)

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        responses = ask(llm, [build_prompt(c, kind) for c in batch], greedy)
        with open(out_path, 'a') as f:
            for config, response in zip(batch, responses):
                f.write(json.dumps({**config, 'label': llm.find_value_in_response(response, 'likelihood')}) + '\n')
        print(f"[{kind}] {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dataset', required=True, help='data/<dataset>.csv')
    parser.add_argument('--filepath', default='./data/')
    parser.add_argument('--max_columns', type=int, default=None, help='column cap of the runs (experiments_env.sh)')
    parser.add_argument('--interesting_attributes_path', default=None,
                        help='default: precompute/<dataset>_interesting_attrs.json')
    parser.add_argument('--kinds', nargs='+', default=KINDS, choices=KINDS)
    parser.add_argument('--n', type=int, default=10000, help='configs to label per kind')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample', action='store_true', help="sample Llama's answers instead of greedy decoding")
    parser.add_argument('--out_dir', default='result/proxy/data')
    parser.add_argument('--transformer_path', default=os.path.expanduser('~/.cache/huggingface'))
    parser.add_argument('--hf_token_path', default='./hf_token.txt')
    args = parser.parse_args()

    for var in ('HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE'):
        os.environ[var] = args.transformer_path
    if os.path.exists(args.hf_token_path) and os.path.getsize(args.hf_token_path) > 0:
        with open(args.hf_token_path) as f:
            os.environ['HF_TOKEN'] = f.read().strip()

    attrs_path = args.interesting_attributes_path or f"precompute/{args.dataset}_interesting_attrs.json"
    with open(attrs_path) as f:
        interesting_attrs = json.load(f)
    df = load_table(args.filepath, args.dataset, args.max_columns)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import set_seed
    from src.utility.llama import Llama3
    set_seed(args.seed)
    llm = Llama3()

    for kind in args.kinds:
        heads, tails = config_space(df, interesting_attrs, kind)
        print(f"[{kind}] {args.dataset}: {len(heads) * len(tails):,} possible configs", flush=True)
        configs = sample_configs(heads, tails, kind, args.n, args.seed)
        label_configs(configs, kind, llm, os.path.join(args.out_dir, f"{args.dataset}_{kind}.jsonl"), args.batch_size,
                      greedy=not args.sample)


if __name__ == '__main__':
    main()
