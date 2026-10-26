"""Train an LLM-proxy decision tree (Section 5.1.2) on generate_dataset.py labels.

    python -m src.prompt_learning.train --data result/proxy/data/Video_ratio.jsonl --kind ratio \
        --out result/proxy/model/Video_10000_ratio_decision_tree.pkl

One-hot features, then a decision tree, fitted on 80% of n seeded samples. Writes the tree and <name>_report.csv.
"""
import argparse
import json
import os
import random

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

LABELS = {'Very Likely', 'Likely', 'Neutral', 'Unlikely', 'Very Unlikely'}   # keys of insightfulness.LIKELIHOOD_MAP

# (feature_names_in_ order, ColumnTransformer order)
TREND_FEATURES = (['index_attr', 'index_values', 'value_attr', 'aggfunc', 'insight_type', 'column_attr', 'bucket'],
                  ['index_attr', 'index_values', 'column_attr', 'value_attr', 'aggfunc', 'insight_type', 'bucket'])
FEATURES = {
    'correlation': TREND_FEATURES,
    'ratio': TREND_FEATURES,
    'outlier': (['index_attr', 'index_value', 'value_attr', 'aggfunc', 'column_attr', 'column_value'],
                ['index_attr', 'index_value', 'column_attr', 'column_value', 'value_attr', 'aggfunc']),
}


def to_features(cfg: dict, kind: str) -> dict:
    """The feature row the runs pass to the tree (pivot_formatter.get_*_clf_prompt_vars)."""
    row = {k: cfg[k] for k in FEATURES[kind][0]}
    if kind != 'outlier':
        row['index_values'] = '|'.join(str(v) for v in sorted(cfg['index_values']))
        row['bucket'] = cfg['bucket'].replace('_', ' ')  # as insightfulness.bucketize_ratio spells it
    else:
        row['index_value'], row['column_value'] = str(cfg['index_value']), str(cfg['column_value'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', required=True, help='JSONL from generate_dataset.py')
    parser.add_argument('--kind', required=True, choices=list(FEATURES))
    parser.add_argument('--n', type=int, default=10000, help='labeled configs to sample (all if fewer)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', required=True, help='output .pkl')
    args = parser.parse_args()

    with open(args.data) as f:
        configs = [json.loads(line) for line in f if line.strip()]
    labeled = [c for c in configs if c.get('label') in LABELS]
    print(f"{args.data}: {len(configs)} configs, {len(labeled)} with a valid label")
    if len(labeled) > args.n:
        labeled = random.Random(args.seed).sample(labeled, args.n)

    X = pd.DataFrame([to_features(c, args.kind) for c in labeled])
    y = [c['label'] for c in labeled]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = make_pipeline(
        ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES[args.kind][1])]),
        DecisionTreeClassifier(criterion='gini', max_depth=15, random_state=42),
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    y_pred = clf.predict(X_test)
    print(f"train accuracy {train_acc:.3f}, test accuracy {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(clf, args.out)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report['overall_train_accuracy'] = {'precision': train_acc, 'recall': train_acc, 'f1-score': train_acc,
                                        'support': len(y_train)}
    pd.DataFrame(report).transpose().to_csv(os.path.splitext(args.out)[0] + '_report.csv')
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()
