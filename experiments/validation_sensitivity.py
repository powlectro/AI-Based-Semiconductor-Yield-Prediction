"""Development-only chronological and repeated-CV diagnostics, no test evaluation.

These are diagnostic experiments: the saved deployment model is unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.secom_io import load_secom
from src.preprocessing import SecomPreprocessor
from src.model_evaluation import evaluate_at_threshold


def experiment(X, y, train_idx, val_idx, k=16):
    prep = SecomPreprocessor(k=k)
    xtr = prep.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
    xval = prep.transform(X.iloc[val_idx])
    model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
    model.fit(xtr, y.iloc[train_idx])
    scores = model.predict_proba(xval)[:, 1]
    yt = y.iloc[val_idx].to_numpy()
    # Fixed 0.5 threshold, not tuned on this validation partition.
    pred = scores >= .5
    return dict(train_n=len(train_idx), validation_n=len(val_idx), validation_failures=int(sum(yt)), recall=float(recall_score(yt, pred, zero_division=0)), average_precision=float(average_precision_score(yt, scores)) if len(np.unique(yt)) == 2 else None, false_alarms=int(sum(pred & (yt == 0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()
    X, y, times = load_secom(ROOT / 'data/raw/secom.zip')
    saved = pd.read_csv(ROOT / 'reports/split_assignments.csv')
    if not np.array_equal(saved.row_index.to_numpy(), np.arange(len(X))):
        raise ValueError('saved row assignments not in original order')
    dev = saved.loc[saved.split != 'test', 'row_index'].to_numpy()
    rows = []
    for fold, (itr, iva) in enumerate(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42).split(X.iloc[dev], y.iloc[dev])):
        rows.append(dict(experiment='stratified_cv', fold=fold, **experiment(X, y, dev[itr], dev[iva])))
    # Chronological experiment on DEVELOPMENT rows only. Test never accessed.
    ordered = dev[np.argsort(times.iloc[dev].to_numpy(), kind='stable')]
    boundary = int(len(ordered) * .75)
    rows.append(dict(experiment='chronological_dev_75_25', fold=0, **experiment(X, y, ordered[:boundary], ordered[boundary:])))
    out = ROOT / 'reports' / 'validation_sensitivity.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('These are development-only diagnostics; no test evaluation.')


if __name__ == '__main__':
    main()
