# AI-Based Semiconductor Yield Prediction with RTL Hardware Implementation

**Course:** Georgia Tech ECE 8803 AI for Semiconductors (Fall 2026)

## Project question
Can we predict a rare fab failure from a small number of process measurements,
then implement a compact classifier in synthesizable fixed-point SystemVerilog?

## Step 1: dataset audit
1. Place these files into the team's existing GitHub repository **without overwriting teammates' existing work**.
2. Windows PowerShell in the Cursor terminal, from the repository root:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe scripts/01_data_audit.py
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python scripts/01_data_audit.py
   .venv/bin/python -m unittest discover -s tests -v
   ```

3. Inspect `reports/data_audit.json`, `reports/feature_statistics.csv`, and PNG plots.
   Open `notebooks/01_secom_data_understanding.ipynb` in Cursor to see the saved notebook walkthrough.
4. Commit the code and generated reports to GitHub.

## Labels and data source
- Original UCI labels: **-1 = pass; +1 = fail**.
- Our code encodes: **0 = pass; 1 = fail**.
- Files are from the UCI SECOM dataset, 2008, CC BY 4.0.
- Citation: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C54305
- https://archive.ics.uci.edu/dataset/179/secom
- **Important:** The actual `secom.data` archive contains **590 sensor columns**;
  some UCI overview text describes 591 features. Trust the parsed data shape.

## Plan and course connection
- Module 2 / FDC: PCA, anomaly scoring, ROC curves.
- Module 3 / classification: logistic regression, random forest, neural network.
- Module 5 / SPC: variation, feature drift, and monitoring.
- Module 6 / virtual metrology: validation strategy, data leakage prevention.
- Module 7 / qualification: false alarms/overkill vs missed failures/escapes.
- CRISP-DM, reproducibility, Gradio walkthrough and demo.

Future milestones: leakage-safe preprocessing; proper held-out tests; compact
feature subset; ML baselines; threshold/cost analysis; quantization; parameter
export; SystemVerilog inference; Python/RTL equivalence; synthesis; demo.

## Important guardrails
- Feature names are generic (`sensor_000`, ...); the original dataset provides
  **no mapping to physical sensor types**, no lot IDs, and no chamber IDs.
- Never claim a causal process explanation or lot-grouped validation from this archive.
- Never use test data to fit an imputer, standardizer, feature selector, threshold,
  class rebalance operation, or model.
- Dataset exploration here is not a deployed predictive capability. Quantified
  yield/cost numbers later will be **scenario estimates**, not real fab savings.



# Step 2 — Leakage-Safe Preprocessing and an Initial Baseline

This **adds** files to the existing Step 1 repository; do not replace existing files.

## Run (Windows PowerShell inside Cursor, from repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/02_prepare_features.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No new dependencies beyond Step 1's `requirements.txt` are needed. The `scikit-learn` dependency installs `joblib`.

## Data methodology

- Original UCI pass/fail labels are mapped to **0=pass** and **1=fail** in `src/secom_io.py`.
- Random, stratified **60% train / 20% validation / 20% test**, seed `42`.
- The SECOM archive has no lot or chamber labels. The random split is an initial baseline, not a guarantee of deployment robustness. Do a separate forward-in-time sensitivity test in a later stage.
- For each `k` in `{8,16,32}`, use **training rows only** to drop sensors with >=50% missing data and train-constant sensors; median impute; remove remaining zero-variance sensors; z-score scale; select the **top k ANOVA-F sensors based on training labels**.
- Transformation of validation rows uses only training-fitted statistics and selections.
- **Test rows must not be used for selecting features/models, adjusting thresholds, or evaluating iteration performance.** They stay sealed for a one-time final assessment.
- Since failures are rare, compare logistic regression (`class_weight="balanced"`) with a naive always-pass model on **validation only**. Step 2 fixes the prediction threshold at `0.50` for comparability. Step 3 will broaden model choice and evaluate cost-aware thresholds on validation only.

## Generated output

| Output file | Purpose |
| --- | --- |
| `reports/split_assignments.csv` | Original row index, assigned split, and timestamp. Reuse these assignments in future stages. |
| `reports/split_summary.json` | Class counts per partition. |
| `reports/preprocessing_summary.json` | Drop counts and precise selected sensor order for each k. |
| `reports/selected_features.csv` | Original sensor name, model-input order, training-only ANOVA-F score, train missingness. |
| `reports/validation_baseline_metrics.csv` | Validation accuracy, recall, FPR, PR-AUC, confusion matrix counts. |
| `reports/figures/validation_recall_vs_false_alarms.png` | Visual manufacturing trade-off. |
| `artifacts/logistic_k08.joblib`, `logistic_k16.joblib`, `logistic_k32.joblib` | Saved preprocessing + logistic model + threshold for reproducibility. **Never load untrusted joblib files.** |
| `notebooks/02_feature_preprocessing.ipynb` | Notebook walkthrough of the generated evidence. Run after the script. |

## Class concepts

- **Module 2 (FDC):** high-dimensional data cleaning and feature selection before fault classification.
- **Module 5 (SPC):** training-only sample statistics and variance. The scaling used here is not itself an SPC control chart or process capability assessment.
- **Module 6 (Virtual Metrology):** data leakage, high-dimensional modeling, limited labeled data, and a need for temporal validation.
- **Module 7 (Test and Qualification):** failure recall vs unnecessary rejection (FP). Define "escape" here as *model-predicted pass for a failing label*, not a proven fab escape.
- **Hardware linkage:** feature count 8/16/32 translates into the number of input values and multiply-accumulate terms in a linear classifier. This is not yet an area/timing synthesis result.

**Important:** ANOVA-F selects sensors correlated with the binary training labels, not necessarily physically causal sensors. False negatives/positives are model errors on SECOM labels; their financial impacts cannot be estimated without explicit fab cost assumptions.

## Next step

Compare logistic regression, random forest, and a small neural network using the **same fixed splits**. Compare average precision, recall, FPR, and scenario cost. Select candidate(s) for fixed-point export and SystemVerilog verification.


## Step 3: ML Model Training and Comparison

This stage compares three machine learning models for predicting semiconductor manufacturing failures using the SECOM dataset.

### Models

- Logistic Regression
- Random Forest
- Small Neural Network (MLP)

Each model is evaluated using 8, 16, and 32 selected sensor features, giving 9 total experiments.

### Methodology

The dataset uses the same stratified 60/20/20 split as Step 2.

- Training: Fit preprocessing and train models.
- Validation: Compare models and select classification thresholds.
- Test: Held out for final evaluation.

Class imbalance is addressed using class weighting for logistic regression and random forest, and training-only oversampling for the neural network.

### Evaluation

Models are compared using:
- Accuracy
- Failure precision and recall
- False positive rate
- F1 score
- Precision-recall AUC (average precision)
- ROC-AUC
- Confusion matrices
- Illustrative manufacturing screening cost

Screening cost = 10 * escaped failures + 1 * false alarms.

The cost weights are hypothetical and are used to study the trade-off between catching bad samples and unnecessarily rejecting good samples.

### Run Step 3

```powershell
.\.venv\Scripts\python.exe scripts/03_train_compare.py
```

### Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Outputs

Reports:
- reports/step3_validation_results.csv
- reports/step3_threshold_sweep.csv
- reports/step3_selection.json

Figures:
- reports/figures/step3_precision_recall.png
- reports/figures/step3_roc_curves.png
- reports/figures/step3_best_confusion_matrix.png

Trained models are saved locally under artifacts/.

### Hardware Direction

The project aims to identify a compact model that can eventually be implemented using fixed-point SystemVerilog.

Logistic regression is an initial hardware candidate because its inference requires a weighted sum and threshold comparison.

The final hardware model will be selected after further evaluation and quantization experiments.

### Limitations

SECOM contains relatively few failing samples, so validation performance is uncertain. Model and threshold selection on the same validation set may be optimistic. The test set remains untouched.

The class-weighted and oversampled models' probability scores are not necessarily calibrated.
```


### Step 3 Results

Nine models were trained using 8, 16, and 32 features.

The best validation results under the illustrative
screening cost (10 × FN + FP) were:

| Model | Features | Failure Recall | False Alarms | Cost |
|---|---|---|---|---|
| Random Forest | 32 | 71.4% | 112 | 172 |
| Logistic Regression | 16 | 57.1% | 85 | 175 |
| Logistic Regression | 8 | 42.9% | 58 | 178 |

Random forest achieved the lowest validation cost,
but logistic regression with 16 features provides
a simpler candidate for hardware implementation.

The screening cost weights are hypothetical.
Results are preliminary because model and threshold
selection used only 21 validation failures.

All nine models were trained successfully.
The test set remains untouched.
