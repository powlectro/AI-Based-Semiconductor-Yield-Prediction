"""Freeze the hardware classifier and reproduce the published validation result.

Fits logistic regression at 8, 16, and 32 features on the saved training split
under the pinned software stack. The held-out test rows are never transformed
and never scored here.

Writes `artifacts/step3_logistic_kXX.joblib` and `reports/freeze.json`.
The freeze record is only written if every logistic validation row matches
`reports/step3_validation_results.csv` exactly. That check authorizes the
one-shot held-out evaluation in `scripts/06_final_test.py`.
"""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

# Keep linear algebra single-threaded before importing NumPy/scikit-learn so
# liblinear takes a deterministic optimization path across machines.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_evaluation import choose_threshold, evaluate_at_threshold
from src.preprocessing import SecomPreprocessor, stratified_split_indices
from src.secom_io import load_secom

# Frozen design parameters chosen solely from Steps 3–5 validation results.
SEED = 42
FEATURE_COUNTS = (8, 16, 32)
FROZEN_FEATURES = 16
FROZEN_BITS = 12
FROZEN_THRESHOLD = 0.50
ESCAPE_COST = 10
FALSE_ALARM_COST = 1


def package_versions():
    """Capture the exact runtime stack to verify against software_environment.json."""
    import sklearn

    versions = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__,
        'joblib': joblib.__version__,
    }
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except ImportError:
        versions['scipy'] = None
    return versions


def load_split(frame_length):
    """Read and validate the saved train/validation/test index partition."""
    assignments = pd.read_csv(ROOT / 'reports' / 'split_assignments.csv')
    if len(assignments) != frame_length or assignments['row_index'].nunique() != frame_length:
        raise SystemExit('split_assignments.csv does not cover the dataset')
    groups = {}
    for name in ('train', 'validation', 'test'):
        groups[name] = assignments.loc[assignments['split'] == name, 'row_index'].to_numpy()
    covered = set(groups['train']) | set(groups['validation']) | set(groups['test'])
    disjoint = (
        len(set(groups['train']) & set(groups['validation'])) == 0
        and len(set(groups['train']) & set(groups['test'])) == 0
        and len(set(groups['validation']) & set(groups['test'])) == 0
    )
    if not disjoint or covered != set(range(frame_length)):
        raise SystemExit('saved split partitions are not a partition of the rows')
    return groups


def coefficient_digest(model):
    """SHA-256 fingerprint of fitted weights and intercept for downstream integrity checks."""
    payload = np.asarray(model.coef_, dtype=np.float64).tobytes()
    payload += np.asarray(model.intercept_, dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def assert_matches_published(record, published):
    """Abort if refitted validation metrics drift from the published Step 3 table."""
    expected = {
        'threshold': float(published['threshold']),
        'tp': int(published['caught_failures_tp']),
        'fn': int(published['escaped_failures_fn']),
        'fp': int(published['false_alarms_fp']),
        'tn': int(published['true_pass_tn']),
        'screening_cost': int(published['screening_cost']),
    }
    got = {
        'threshold': float(record['threshold']),
        'tp': int(record['tp']),
        'fn': int(record['fn']),
        'fp': int(record['fp']),
        'tn': int(record['tn']),
        'screening_cost': int(record['screening_cost']),
    }
    if got != expected:
        raise SystemExit(
            f"k={record['features']} validation confusion does not match the published table.\n"
            f'got      {got}\n'
            f'expected {expected}'
        )
    for key in ('pr_auc', 'roc_auc'):
        if abs(float(record[key]) - float(published[key])) > 5e-5:
            raise SystemExit(
                f"k={record['features']} {key} {record[key]} != published {published[key]}"
            )


def main():
    # Guardrail: once both the local artifact and the held-out test report exist,
    # refuse to overwrite the model so the classifier cannot be silently retuned.
    # On a fresh clone (where gitignored artifacts/ is empty), this still allows
    # recreating the exact frozen joblib bundles from the pinned environment.
    test_report = ROOT / 'reports' / 'step6_test_results.json'
    frozen_artifact = ROOT / 'artifacts' / 'step3_logistic_k16.joblib'
    if test_report.is_file() and frozen_artifact.is_file():
        raise SystemExit(
            f'{test_report.name} and {frozen_artifact.name} already exist. '
            'Refitting would replace the classifier after the held-out score was recorded.'
        )

    X, y, _timestamps = load_secom(ROOT / 'data' / 'raw' / 'secom.zip')
    groups = load_split(len(X))
    fresh = stratified_split_indices(y, random_state=SEED)

    # Confirm the saved CSV split has the exact same partition membership as Step 2.
    saved_ok = (
        np.array_equal(np.sort(groups['train']), np.sort(fresh.train))
        and np.array_equal(np.sort(groups['validation']), np.sort(fresh.validation))
        and np.array_equal(np.sort(groups['test']), np.sort(fresh.test))
    )
    if not saved_ok:
        raise SystemExit('split_assignments.csv does not match stratified_split_indices(seed=42)')

    # Use the original stratified_split_indices row order: liblinear processes
    # rows sequentially, whereas the CSV is sorted by row_index and would shift coefficients.
    groups = {'train': fresh.train, 'validation': fresh.validation, 'test': fresh.test}

    published = pd.read_csv(ROOT / 'reports' / 'step3_validation_results.csv')
    published = published.loc[published['model'] == 'logistic'].set_index('features')
    feature_summary = json.loads((ROOT / 'reports' / 'preprocessing_summary.json').read_text(encoding='utf-8'))
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)

    # Fit train-only preprocessing and logistic models for k in {8, 16, 32}.
    # Nothing below touches groups['test'].
    trained = {}
    for k in FEATURE_COUNTS:
        prep = SecomPreprocessor(k=k, max_missing_fraction=0.50)
        train_x = prep.fit_transform(X.iloc[groups['train']], y.iloc[groups['train']].to_numpy())
        val_x = prep.transform(X.iloc[groups['validation']])
        y_val = y.iloc[groups['validation']].to_numpy()

        # Verify selected sensor order matches the published Step 2 hardware input order.
        expected_names = feature_summary['feature_counts'][str(k)]['selected_sensor_input_order']
        if list(prep.selected_features_) != expected_names:
            raise SystemExit(f'k={k} selected sensors do not match preprocessing_summary.json')

        model = LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=SEED,
        )
        model.fit(train_x, y.iloc[groups['train']].to_numpy())
        scores = model.predict_proba(val_x)[:, 1]
        selected, _sweep = choose_threshold(y_val, scores, ESCAPE_COST, FALSE_ALARM_COST)
        checked = evaluate_at_threshold(y_val, scores, float(selected['threshold']), ESCAPE_COST, FALSE_ALARM_COST)
        record = {
            'features': k,
            'threshold': checked['threshold'],
            'tp': checked['tp'],
            'fn': checked['fn'],
            'fp': checked['fp'],
            'tn': checked['tn'],
            'screening_cost': checked['screening_cost'],
            'pr_auc': checked['pr_auc'],
            'roc_auc': checked['roc_auc'],
            'failure_recall': checked['failure_recall'],
            'false_positive_rate': checked['false_positive_rate'],
        }
        assert_matches_published(record, published.loc[k])

        # Bundle preprocessor, model, threshold, and weight hash together for Steps 6–7.
        bundle = {
            'preprocessor': prep,
            'model': model,
            'threshold': float(selected['threshold']),
            'selected_features': list(prep.selected_features_),
            'cost_assumptions': {'escape': ESCAPE_COST, 'false_alarm': FALSE_ALARM_COST},
            'coefficient_sha256': coefficient_digest(model),
        }
        path = artifacts / f'step3_logistic_k{k:02d}.joblib'
        joblib.dump(bundle, path)
        trained[k] = bundle
        print(
            f"logistic k={k:02d} threshold={record['threshold']:.2f} "
            f"recall={record['failure_recall']:.3f} fp={record['fp']} "
            f"cost={record['screening_cost']} matches published validation"
        )

    frozen = trained[FROZEN_FEATURES]
    if abs(frozen['threshold'] - FROZEN_THRESHOLD) > 1e-9:
        raise SystemExit(
            f"Published k=16 threshold is {frozen['threshold']}, not the frozen {FROZEN_THRESHOLD}"
        )

    # Confirm the current Python/library versions match the published software record.
    versions = package_versions()
    recorded = json.loads((ROOT / 'reports' / 'software_environment.json').read_text(encoding='utf-8'))
    for key in ('python', 'numpy', 'pandas', 'scikit_learn', 'joblib'):
        if versions[key] != recorded[key]:
            raise SystemExit(
                f'Environment {key} is {versions[key]}, published record is {recorded[key]}'
            )

    # Document why logistic regression (16x12) was chosen for RTL over the software-only forest.
    forest = json.loads((ROOT / 'reports' / 'step3_selection.json').read_text(encoding='utf-8'))
    best = forest['best_validation_model']
    freeze = {
        'chosen_classifier': {
            'family': 'logistic_regression',
            'features': FROZEN_FEATURES,
            'bits': FROZEN_BITS,
            'threshold': FROZEN_THRESHOLD,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'random_state': SEED,
            'artifact': 'artifacts/step3_logistic_k16.joblib',
            'coefficient_sha256': frozen['coefficient_sha256'],
            'selected_features': frozen['selected_features'],
            'reason': (
                'Inference is a weighted sum plus a threshold, which is the RTL datapath. '
                'On the published validation table this point has screening cost 175 and, '
                'at 12 bits, agrees with every floating-point validation decision.'
            ),
        },
        'not_selected_for_hardware': {
            'model': best['model'],
            'features': best['features'],
            'threshold': best['threshold'],
            'validation_screening_cost': best['screening_cost'],
            'reason': (
                'Lower illustrative validation cost than the logistic model, but a forest '
                'is not the fixed-point linear classifier that was synthesized.'
            ),
        },
        'environment': versions,
        'environment_matches_software_environment_json': True,
        'validation_matches_published_step3': True,
        'validation_k16': {
            'tp': 12,
            'fn': 9,
            'fp': 85,
            'tn': 208,
            'failure_recall': 12 / 21,
            'false_positive_rate': 85 / 293,
            'screening_cost': 175,
        },
        'cost_assumptions': {
            'escaped_failure': ESCAPE_COST,
            'false_alarm': FALSE_ALARM_COST,
            'note': 'Illustrative costs, not fab prices.',
        },
        'prohibitions': [
            'Do not change the model, feature count, bit width, or threshold after reading the test score.',
            'Do not refit the imputer, scaler, selector, or classifier on test rows.',
        ],
    }
    freeze_path = ROOT / 'reports' / 'freeze.json'
    freeze_path.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(f'Saved: {freeze_path}')
    print('Validation reproduced the published logistic rows. TEST was not scored.')


if __name__ == '__main__':
    main()
