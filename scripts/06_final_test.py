"""Score the frozen classifier on the held-out test split, once.

Reads `reports/freeze.json` and `artifacts/step3_logistic_k16.joblib`.
Does not fit, does not sweep thresholds, and does not write a new model.
Also exports the quantized test vectors and verifies both the parallel and
sequential RTL modules against Python in Icarus Verilog.
Re-running reprints the same frozen score; it is not a second model selection.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fixed_point import make_fixed_model, quantize, quantized_linear_predict
from src.model_evaluation import evaluate_at_threshold
from src.secom_io import load_secom

ESCAPE_COST = 10
FALSE_ALARM_COST = 1


def require_tool(name):
    """Fail fast with an actionable message if Icarus Verilog is not on PATH."""
    if not shutil.which(name):
        raise SystemExit(
            f'{name} was not found on PATH. Activate the frozen conda env '
            '(conda activate secom-freeze) or install Icarus Verilog.'
        )


def load_export():
    """Import Step 4 helpers dynamically (numeric filename cannot use regular import)."""
    spec = importlib.util.spec_from_file_location(
        'export_fixed_point', ROOT / 'scripts' / '04_export_fixed_point.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confusion_from_bits(y_true, predicted):
    """Compute confusion counts and screening metrics from hard binary predictions."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(predicted & (y_true == 1)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    return {
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn,
        'accuracy': (tp + tn) / len(y_true),
        'failure_recall': tp / (tp + fn) if (tp + fn) else 0.0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) else 0.0,
        'failure_precision': tp / (tp + fp) if (tp + fp) else 0.0,
        'screening_cost': ESCAPE_COST * fn + FALSE_ALARM_COST * fp,
    }


def simulate(folder, samples, features, bits, kind):
    """Compile and run either the parallel wrapper or sequential MAC testbench on test vectors."""
    if kind == 'parallel':
        cmd = [
            'iverilog', '-g2012', '-s', 'classifier_top_tb',
            f'-DBITS={bits}', f'-DN={features}', f'-DSAMPLES={samples}',
            '-I', str(folder), '-o', str(folder / 'classifier_top_sim'),
            str(ROOT / 'hardware' / 'linear_classifier.sv'),
            str(ROOT / 'hardware' / 'linear_classifier_top.sv'),
            str(ROOT / 'hardware' / 'classifier_top_tb.sv'),
        ]
        expect = f'PASS: {samples} Python/RTL vector comparisons, latency_cycles=1'
    else:
        cmd = [
            'iverilog', '-g2012', '-s', 'classifier_seq_tb',
            f'-DBITS={bits}', f'-DN={features}', f'-DSAMPLES={samples}',
            '-I', str(folder), '-o', str(folder / 'classifier_seq_sim'),
            str(ROOT / 'hardware' / 'linear_classifier_seq.sv'),
            str(ROOT / 'hardware' / 'classifier_seq_tb.sv'),
        ]
        expect = (
            f'PASS: {samples} Python/RTL vector comparisons, '
            f'latency_cycles={features}, interval_cycles={features}'
        )
    subprocess.run(cmd, cwd=folder, check=True)
    binary = 'classifier_top_sim' if kind == 'parallel' else 'classifier_seq_sim'
    run = subprocess.run(['vvp', str(folder / binary)], cwd=folder, check=True, capture_output=True, text=True)
    print(run.stdout.strip())
    if expect not in run.stdout:
        raise SystemExit(f'{kind} RTL did not match the frozen test vectors')
    return 'passed'


def markdown(report):
    """Format the held-out test metrics and caveats into a standalone Markdown report."""
    floated = report['float']
    fixed = report['fixed_point']
    lines = [
        '# Held-out test evaluation',
        '',
        'The classifier, feature count, bit width, and threshold were frozen in',
        '`reports/freeze.json` before this file was written. This score did not',
        'change them.',
        '',
        f"Model: logistic regression, {report['features']} features, threshold {report['threshold']:.2f}.",
        f"Fixed point: {report['bits']} bits "
        f"({report['input_fractional_bits']} input fractional bits, "
        f"{report['weight_fractional_bits']} weight fractional bits).",
        f"Test rows: {report['samples']} ({report['failures']} failures, {report['passes']} passes).",
        '',
        '## Prediction metrics',
        '',
        '| Decision | Recall | False-positive rate | Precision | Escapes | False alarms | Screening cost | Agreement with float |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        (
            f"| Floating point | {floated['failure_recall']:.4f} | {floated['false_positive_rate']:.4f} | "
            f"{floated['failure_precision']:.4f} | {floated['fn']} | {floated['fp']} | "
            f"{floated['screening_cost']} | 1.0000 |"
        ),
        (
            f"| {report['bits']}-bit | {fixed['failure_recall']:.4f} | {fixed['false_positive_rate']:.4f} | "
            f"{fixed['failure_precision']:.4f} | {fixed['fn']} | {fixed['fp']} | "
            f"{fixed['screening_cost']} | {report['agreement_with_float']:.4f} |"
        ),
        '',
        f"Floating-point average precision (PR-AUC) {floated['pr_auc']:.4f}, "
        f"ROC-AUC {floated['roc_auc']:.4f}, accuracy {floated['accuracy']:.4f}.",
        f"Confusion, floating point: TP {floated['tp']}, FN {floated['fn']}, "
        f"FP {floated['fp']}, TN {floated['tn']}.",
        f"Parallel RTL on these test vectors: {report['parallel_rtl']}.",
        f"Sequential MAC RTL on these test vectors: {report['sequential_rtl']}.",
        '',
        'Screening cost = 10 × escaped failures + 1 × false alarms. '
        'The weights are the same illustration used on the validation set. They are not fab prices.',
        '',
        '## Limitations',
        '',
    ]
    for item in report['limitations']:
        lines.append(f'- {item}')
    lines.append('')
    return '\n'.join(lines)


def main():
    require_tool('iverilog')
    require_tool('vvp')

    # 1. Verify the freeze record and check that the local joblib matches it exactly.
    freeze_path = ROOT / 'reports' / 'freeze.json'
    if not freeze_path.is_file():
        raise SystemExit('reports/freeze.json is missing. Run scripts/06_freeze_classifier.py first.')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    chosen = freeze['chosen_classifier']
    if not freeze.get('validation_matches_published_step3'):
        raise SystemExit('Freeze record did not reproduce the published validation table.')

    artifact = ROOT / chosen['artifact']
    if not artifact.is_file():
        raise SystemExit(f'{artifact} is missing. Run scripts/06_freeze_classifier.py first.')
    bundle = joblib.load(artifact)  # locally generated, trusted artifact only
    if bundle.get('coefficient_sha256') != chosen['coefficient_sha256']:
        raise SystemExit('Artifact coefficients do not match the freeze record.')
    if int(bundle['model'].coef_.shape[1]) != int(chosen['features']):
        raise SystemExit('Artifact feature count does not match the freeze record.')
    if abs(float(bundle['threshold']) - float(chosen['threshold'])) > 1e-9:
        raise SystemExit('Artifact threshold does not match the freeze record.')

    # 2. Transform the sealed test split using the train-fitted preprocessor only.
    features = int(chosen['features'])
    bits = int(chosen['bits'])
    threshold = float(chosen['threshold'])
    X, y, _timestamps = load_secom(ROOT / 'data' / 'raw' / 'secom.zip')
    export = load_export()
    split = export.load_indices(ROOT / 'reports' / 'split_assignments.csv', len(X))
    test_x = bundle['preprocessor'].transform(X.iloc[split['test']])
    y_test = y.iloc[split['test']].to_numpy()

    # Fixed-point input/weight scaling is derived strictly from training rows.
    train_x = bundle['preprocessor'].transform(X.iloc[split['train']])
    spec = make_fixed_model(bundle['model'], train_x, bits, threshold)

    # 3. Evaluate both floating-point and 12-bit integer decisions at the frozen threshold.
    scores = bundle['model'].predict_proba(test_x)[:, 1]
    floated = evaluate_at_threshold(y_test, scores, threshold, ESCAPE_COST, FALSE_ALARM_COST)
    x_int = quantize(test_x, bits, spec['xfrac'])
    fixed_pred = quantized_linear_predict(x_int, spec['weights'], spec['offset'])
    fixed = confusion_from_bits(y_test, fixed_pred)
    agreement = float(np.mean((scores >= threshold) == fixed_pred))

    # 4. Export test vectors and confirm bit-exact RTL agreement on both architectures.
    folder = ROOT / 'hardware' / 'generated' / f'k{features:02d}_b{bits:02d}_test'
    folder.mkdir(parents=True, exist_ok=True)
    export.write_rtl_constants(folder, spec)
    export.write_vectors(folder, x_int, fixed_pred, bits)
    (folder / 'config.json').write_text(json.dumps({
        **spec,
        'features': bundle['selected_features'],
        'partition': 'test',
        'rows': len(test_x),
    }, indent=2) + '\n', encoding='utf-8')

    parallel = simulate(folder, len(test_x), features, bits, 'parallel')
    sequential = simulate(folder, len(test_x), features, bits, 'sequential')

    # 5. Write structured JSON and human-readable Markdown reports with explicit caveats.
    limitations = [
        f'The test split has {int(np.sum(y_test == 1))} failures. A recall difference of one wafer moves the rate by about {1 / max(int(np.sum(y_test == 1)), 1):.1%}.',
        'Feature count, bit width, and threshold were chosen on the validation split. This test score can look worse than the validation score, and it must not be used to pick a new one.',
        'Class-weighted logistic scores are not calibrated probabilities.',
        'Screening cost uses illustrative weights of 10 per escaped failure and 1 per false alarm. It is not a fab cost.',
        'Sensor names are anonymous. A selected column is not a demonstrated physical cause.',
        'The RTL sees train-median-imputed, standardized, selected features. Raw-sensor preprocessing is outside the chip.',
        'Python/RTL agreement on this file checks the frozen integer decision. It is not a new accuracy claim beyond the counts above.',
        'Nangate45 area and timing live in the hardware reports. They have no wire-load model, no clock tree, and no power number.',
    ]
    report = {
        'partition': 'test',
        'samples': int(len(y_test)),
        'failures': int(np.sum(y_test == 1)),
        'passes': int(np.sum(y_test == 0)),
        'features': features,
        'bits': bits,
        'threshold': threshold,
        'input_fractional_bits': spec['xfrac'],
        'weight_fractional_bits': spec['wfrac'],
        'coefficient_sha256': chosen['coefficient_sha256'],
        'float': floated,
        'fixed_point': fixed,
        'agreement_with_float': agreement,
        'parallel_rtl': parallel,
        'sequential_rtl': sequential,
        'used_for_model_selection': False,
        'limitations': limitations,
    }
    json_path = ROOT / 'reports' / 'step6_test_results.json'
    md_path = ROOT / 'reports' / 'step6_test_results.md'
    json_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    md_path.write_text(markdown(report), encoding='utf-8')
    print(f'Saved: {json_path}')
    print(f'Saved: {md_path}')
    print(
        f"TEST float recall={floated['failure_recall']:.3f} "
        f"fpr={floated['false_positive_rate']:.3f} cost={floated['screening_cost']} "
        f"fixed_agreement={agreement:.3f}"
    )
    print('Threshold, features, and bit width were not changed.')


if __name__ == '__main__':
    main()
