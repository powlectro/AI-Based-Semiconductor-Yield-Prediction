"""Development-only feature-selection frequency over stratified CV folds."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.secom_io import load_secom
from src.preprocessing import SecomPreprocessor


def main():
    X, y, _ = load_secom(ROOT / 'data/raw/secom.zip')
    splits = pd.read_csv(ROOT / 'reports/split_assignments.csv')
    if not np.array_equal(splits.row_index.to_numpy(), np.arange(len(X))):
        raise ValueError('split indexes must match dataset order')
    dev = splits.loc[splits.split != 'test', 'row_index'].to_numpy()
    freq = Counter()
    folds = 5
    for train, _ in StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(X.iloc[dev], y.iloc[dev]):
        prep = SecomPreprocessor(k=16)
        prep.fit(X.iloc[dev[train]], y.iloc[dev[train]])
        freq.update(prep.selected_features_)
    result = pd.DataFrame([{'feature': col, 'selected_folds': n, 'selection_fraction': n/folds} for col, n in freq.most_common()])
    output = ROOT / 'reports/feature_stability.csv'
    result.to_csv(output, index=False)
    print(result.head(20).to_string(index=False))
    print(f'Saved {output}. Only development rows used.')


if __name__ == '__main__':
    main()
