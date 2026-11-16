"""Validation-only Gradio sample demonstration (optional: pip install gradio)."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.secom_io import load_secom
from src.fixed_point import make_fixed_model, quantize, quantized_linear_predict

artifact = ROOT / 'artifacts/step3_logistic_k16.joblib'
if not artifact.exists():
    raise SystemExit('Run python scripts/03_train_compare.py first to create the local model.')
bundle = joblib.load(artifact)  # trusted, local artifact only
X, y, _ = load_secom(ROOT / 'data/raw/secom.zip')
split = pd.read_csv(ROOT / 'reports/split_assignments.csv')
validation_rows = split.loc[split['split'] == 'validation', 'row_index'].to_numpy()
train_rows = split.loc[split['split'] == 'train', 'row_index'].to_numpy()
prep, model = bundle['preprocessor'], bundle['model']
train_x = prep.transform(X.iloc[train_rows])
specs = {b: make_fixed_model(model, train_x, b, float(bundle['threshold'])) for b in (8, 12, 16)}


def predict(validation_position, threshold):
    idx = int(validation_rows[int(validation_position)])
    input_x = prep.transform(X.iloc[[idx]])
    score = float(model.predict_proba(input_x)[0, 1])
    quantized = {}
    for b, spec in specs.items():
        qx = quantize(input_x, b, spec['xfrac'])
        quantized[str(b) + '-bit'] = bool(quantized_linear_predict(qx, spec['weights'], spec['offset'])[0])
    return {
        'Dataset row': idx,
        'Reference label': 'Fail' if int(y.iloc[idx]) else 'Pass',
        'Float failure score': round(score, 5),
        'User-adjusted float prediction': 'Fail' if score >= threshold else 'Pass',
        'Original model threshold': bundle['threshold'],
        'Fixed-point predictions at ORIGINAL trained threshold': quantized,
        'Note': 'Quantized predictions use the trained threshold, not the interactive slider. Validation data only.'
    }

with gr.Blocks(title='SECOM Yield Screening Demo') as demo:
    gr.Markdown('# SECOM Yield Screening — development demonstration\nValidation samples only. No test data or real fab deployment.')
    with gr.Row():
        index = gr.Slider(0, len(validation_rows) - 1, value=0, step=1, label='Validation sample index')
        threshold = gr.Slider(0.05, 0.95, value=float(bundle['threshold']), step=0.05, label='Interactive floating-point threshold')
    output = gr.JSON(label='Prediction and numerical precision comparison')
    gr.Button('Predict').click(predict, inputs=[index, threshold], outputs=output)

if __name__ == '__main__':
    demo.launch()
