# AI-Based Semiconductor Yield Prediction with RTL Hardware Implementation

**Course:** Georgia Tech ECE 8803 AI for Semiconductors (Fall 2026)

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Setup and environment](#setup-and-environment)
- [Step 1: Dataset audit](#step-1-dataset-audit)
- [Step 2: Leakage-safe preprocessing](#step-2-leakage-safe-preprocessing)
- [Step 3: Model comparison](#step-3-model-comparison)
- [Step 4: Fixed-point export and RTL simulation](#step-4-fixed-point-export-and-rtl-simulation)
- [Step 5: Clocked wrapper and Yosys area/timing](#step-5-clocked-wrapper-and-yosys-areatiming)
- [Tests](#tests)
- [Results](#results)
- [Optional demo](#optional-demo)
- [Caveats and next steps](#caveats-and-next-steps)

## Overview

### Project question

Can we predict a rare fab failure from a small number of process measurements,
then implement a compact classifier in synthesizable fixed-point SystemVerilog?

Steps 4 and 5 extend Steps 1–3. They do not replace them. **Do not evaluate the held-out test set until feature count, precision, and threshold are frozen.**

### Labels and data source

- Original UCI labels: **-1 = pass; +1 = fail**.
- Our code encodes: **0 = pass; 1 = fail** in `src/secom_io.py`.
- Files are from the UCI SECOM dataset, 2008, CC BY 4.0.
- Citation: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C54305
- https://archive.ics.uci.edu/dataset/179/secom
- **Important:** The actual `secom.data` archive contains **590 sensor columns**;
  some UCI overview text describes 591 features. Trust the parsed data shape.

### Plan and course connection

- Module 2 / FDC: PCA, anomaly scoring, ROC curves.
- Module 3 / classification: logistic regression, random forest, neural network.
- Module 5 / SPC: variation, feature drift, and monitoring.
- Module 6 / virtual metrology: validation strategy, data leakage prevention.
- Module 7 / qualification: false alarms/overkill vs missed failures/escapes.
- CRISP-DM, reproducibility, Gradio walkthrough and demo.

The original milestone list was: leakage-safe preprocessing; proper held-out tests; compact
feature subset; ML baselines; threshold/cost analysis; quantization; parameter
export; SystemVerilog inference; Python/RTL equivalence; synthesis; demo.
Steps 1–5 below cover that list through a Nangate45 area/timing estimate.
The held-out test and a placed-and-routed design are still open. See [Caveats and next steps](#caveats-and-next-steps).

### Guardrails

- Feature names are generic (`sensor_000`, ...). The original dataset provides
  **no mapping to physical sensor types**, no lot IDs, and no chamber IDs.
- Never claim a causal process explanation or lot-grouped validation from this archive.
- Never use test data to fit an imputer, standardizer, feature selector, threshold,
  class rebalance operation, or model.
- Dataset exploration is not a deployed predictive capability. Quantified
  yield/cost numbers are **scenario estimates**, not real fab savings.
- ANOVA-F selects sensors correlated with the binary training labels, not necessarily physically causal sensors. False negatives and false positives are model errors on SECOM labels. Their financial impact cannot be estimated without explicit fab cost assumptions.
- **Never load untrusted joblib files.**

## Repository layout

| Path | Role |
| --- | --- |
| `src/secom_io.py` | Load the UCI zip. Map labels. No fitting. |
| `src/preprocessing.py` | Train-only split, impute, scale, and ANOVA-F selection. |
| `src/model_evaluation.py` | Recall, false-positive rate, screening cost, threshold choice, train-only oversampling. |
| `src/fixed_point.py` | Bit-accurate signed fixed-point linear inference. |
| `scripts/01_data_audit.py` … `05_hardware_cost.py` | One script per pipeline stage. |
| `experiments/` | Development-only cross-validation, chronological split, and feature-stability checks. |
| `hardware/linear_classifier.sv` | Combinational fixed-point classifier. |
| `hardware/classifier_tb.sv` | Combinational RTL vs Python vectors. |
| `hardware/linear_classifier_top.sv` | 1-cycle clocked wrapper. Does not change the combinational classifier. |
| `hardware/classifier_top_tb.sv` | Wrapper vs the same `vectors.mem` files. |
| `notebooks/` | Saved walkthroughs for Steps 1 and 2. |
| `dashboard/app.py` | Optional Gradio demo. Validation rows only. |
| `tests/` | Python unit tests. |
| `data/raw/secom.zip` | UCI archive. |
| `reports/` | CSV, JSON, and figures. Step 5 also writes `reports/step5_hardware_cost.md`. |
| `artifacts/*.joblib` | Local trained models. Gitignored. |
| `hardware/generated/` | Exported constants and vectors. Gitignored. |
| `hardware/lib/*.lib` | Downloaded Nangate45 liberty file. Gitignored. |
| `hardware/synth/` | Yosys logs and netlists. Gitignored. |

Generated RTL vectors and trained `.joblib` files are intentionally gitignored. Commit source code and reports, not machine-specific binaries.

## Setup and Environment

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Windows, without activating the venv:

```bash
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux, without activating the venv:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

`requirements.txt` covers NumPy, pandas, Matplotlib, scikit-learn, ipykernel, and joblib. No extra packages are required for Steps 2–4. The scikit-learn install provides joblib. Gradio is optional and installed separately for the demo. Yosys is not a Python package. In the course conda environment:

```bash
conda install -c conda-forge yosys
```

Icarus Verilog (`iverilog` and `vvp`) must be on `PATH` for RTL simulation. Install it separately or use a CI runner that already has it.

The dependency pins are open lower bounds. Rerunning Step 3 under a newer scikit-learn can change fitted coefficients and the selected validation threshold. Record package versions alongside published results, and freeze a lockfile before the final evaluation. Do **not** silently compare figures from runs under different library versions. `reports/software_environment.json` is the record from the run that produced the saved reports.

## Step 1: Dataset audit

Exploratory only. This script does not fit a model or a scaler.

Windows:

```bash
.\.venv\Scripts\python.exe scripts/01_data_audit.py
```

macOS/Linux:

```bash
.venv/bin/python scripts/01_data_audit.py
```

Inspect `reports/data_audit.json`, `reports/feature_statistics.csv`, and the PNG plots under `reports/figures/`. Open `notebooks/01_secom_data_understanding.ipynb` for the saved walkthrough.

Published audit: 1,567 samples, 590 sensors, 1,463 pass, 104 fail (6.637% failures), 4.538% of cells missing, timestamps from 2008-07-19 to 2008-10-17.

## Step 2: Leakage-safe preprocessing

This adds files to the Step 1 tree. It does not replace existing files. Only **train** rows fit preprocessors and models. **Validation** is for comparison. **Test** stays sealed.

Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/02_prepare_features.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Data Methodology

- Random, stratified **60% train / 20% validation / 20% test**, seed `42`.
- The SECOM archive has no lot or chamber labels. The random split is an initial baseline, not a guarantee of deployment robustness. A forward-in-time check is in [Step 4](#sensitivity-and-feature-stability).
- For each `k` in `{8,16,32}`, use **training rows only** to drop sensors with >=50% missing data and train-constant sensors; median impute; remove remaining zero-variance sensors; z-score scale; select the **top k ANOVA-F sensors based on training labels**.
- Transformation of validation rows uses only training-fitted statistics and selections.
- **Test rows must not be used for selecting features, models, or thresholds, or for judging iteration performance.** They stay sealed for a one-time final assessment.
- Because failures are rare, compare logistic regression (`class_weight="balanced"`) with a naive always-pass model on **validation only**. Step 2 fixes the prediction threshold at `0.50`. Step 3 widens the model set and sweeps cost-aware thresholds on validation only.

### Generated Output

| Output file | Purpose |
| --- | --- |
| `reports/split_assignments.csv` | Original row index, assigned split, and timestamp. Reuse these assignments in later stages. |
| `reports/split_summary.json` | Class counts per partition. |
| `reports/preprocessing_summary.json` | Drop counts and selected sensor order for each k. |
| `reports/selected_features.csv` | Sensor name, model-input order, training-only ANOVA-F score, train missingness. |
| `reports/validation_baseline_metrics.csv` | Validation accuracy, recall, false-positive rate, PR-AUC, confusion counts. |
| `reports/figures/validation_recall_vs_false_alarms.png` | Recall versus false alarms. |
| `artifacts/logistic_k08.joblib`, `logistic_k16.joblib`, `logistic_k32.joblib` | Preprocessing, logistic model, and threshold. |
| `notebooks/02_feature_preprocessing.ipynb` | Notebook walkthrough. Run it after the script. |

On this split the sizes are 939 / 314 / 314 rows, with 62 / 21 / 21 failures. At the fixed 0.50 threshold, always-pass validation accuracy is about 0.933 with **zero** failure recall. Logistic regression at 16 features has failure recall 0.571 and false-positive rate 0.290 (9 missed failures, 85 false alarms).

### Class Concepts

- **Module 2 (FDC):** high-dimensional cleaning and feature selection before fault classification.
- **Module 5 (SPC):** training-only sample statistics and variance. This scaling is not an SPC control chart or a process-capability study.
- **Module 6 (Virtual Metrology):** leakage, high-dimensional modeling, limited labels, and a need for temporal validation.
- **Module 7 (Test and Qualification):** failure recall versus unnecessary rejection. "Escape" here means *model-predicted pass for a failing label*, not a proven fab escape.
- **Hardware linkage:** feature counts 8, 16, and 32 are the number of inputs and multiply-accumulate terms in the linear classifier. That count is not itself an area or timing result. Step 5 measures those.

Step 3 compares logistic regression, random forest, and a small neural network on the **same fixed splits**, using average precision, recall, false-positive rate, and scenario cost, then picks a candidate for fixed-point export.

## Step 3: Model comparison

Three models, each at 8, 16, and 32 features (nine experiments):

- Logistic regression
- Random forest
- Small neural network (one hidden layer, 16 units)

Same stratified 60/20/20 split as Step 2.

- Training: fit preprocessing and models.
- Validation: compare models and choose thresholds.
- Test: held out.

Class imbalance uses class weights for logistic regression and the forest, and training-only oversampling for the network.

Metrics: accuracy, failure precision and recall, false-positive rate, F1, precision-recall AUC (average precision), ROC-AUC, confusion matrices, and an illustrative screening cost.

Screening cost = 10 × escaped failures + 1 × false alarms.

The weights are hypothetical. They study the tradeoff between catching bad samples and rejecting good ones. They are not fab prices.

Windows:

```bash
.\.venv\Scripts\python.exe scripts/03_train_compare.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Reports:

- `reports/step3_validation_results.csv`
- `reports/step3_threshold_sweep.csv`
- `reports/step3_selection.json`

Figures:

- `reports/figures/step3_precision_recall.png`
- `reports/figures/step3_roc_curves.png`
- `reports/figures/step3_best_confusion_matrix.png`

Trained bundles are saved under `artifacts/` as `step3_<model>_k<features>.joblib`.

Logistic regression is the hardware candidate because inference is a weighted sum and a threshold. The saved Step 3 summary under scikit-learn 1.8.0 selected **16-feature logistic regression**. An earlier summary, and a rerun under scikit-learn 1.9.1, selected a **32-feature random forest** at a slightly lower illustrative cost (172 versus 175). The forest is not the RTL model. Do not compare those tables without checking `reports/software_environment.json`.

A published validation snapshot under the illustrative cost:

| Model | Features | Failure recall | False alarms | Cost |
| --- | ---: | ---: | ---: | ---: |
| Random forest | 32 | 71.4% | 112 | 172 |
| Logistic regression | 16 | 57.1% | 85 | 175 |
| Logistic regression | 8 | 42.9% | 58 | 178 |

SECOM has few failing samples, so validation performance is uncertain. Choosing the model and the threshold on the same validation set can be optimistic. The test set stays untouched. Class-weighted and oversampled probability scores are not necessarily calibrated. All nine models train. The network can hit its iteration cap (`ConvergenceWarning` at `max_iter=300`) and still write a score. That score is unfinished training, not a fully converged network.

## Step 4: Fixed-point export and RTL simulation

### What this stage adds

| Addition | Files | What it does |
| --- | --- | --- |
| Fixed-point reference | `src/fixed_point.py` | Signed integers, rounding, clipping, and an integer linear decision. |
| Model export | `scripts/04_export_fixed_point.py` | Writes 8-, 12-, and 16-bit logistic coefficients, config, and validation vectors. |
| Combinational RTL | `hardware/linear_classifier.sv` | Integer weighted sum and compare. Sensors are already preprocessed. |
| RTL testbench | `hardware/classifier_tb.sv` | Checks RTL against the Python fixed-point vectors in Icarus Verilog. |
| Quantization table | `reports/step4_quantization.csv` | Agreement with floating point, recall, false alarms, and screening cost. |
| Sensitivity | `experiments/validation_sensitivity.py` | Five-fold stratified CV and a chronological split, development rows only. |
| Feature stability | `experiments/feature_stability.py` | How often each of 16 sensors is selected across five training folds. |
| Optional demo | `dashboard/app.py` | Gradio page for validation examples. Install Gradio separately. |
| Tests | `tests/test_fixed_point.py` | Integer encoding, saturation, and the decision boundary. |

The Python reference and the RTL both see **train-median-imputed, standardized, selected** sensors, not raw wafer measurements. Feature order lives in the local joblib artifact. Preprocessing stays outside the hardware.

Run Step 3 before Step 4. The export reads the saved split and `artifacts/step3_logistic_k16.joblib`. It does not fit another model and it does not inspect the test partition.

```bash
python scripts/03_train_compare.py
python scripts/04_export_fixed_point.py --features 16 --bits 8 12 16
python experiments/validation_sensitivity.py
python experiments/feature_stability.py
```

Outputs for the default 16-feature export:

- `hardware/generated/k16_b08/`, `k16_b12/`, and `k16_b16/` (gitignored)
- `model_constants.svh`: signed quantized coefficients and the comparison offset
- `vectors.mem`: validation vectors and the bit-accurate expected prediction. The expected bit is the MSB. Sensor zero occupies the LSB.
- `config.json`: input and weight fractional bits, ordered sensor names, and the threshold
- `reports/step4_quantization.csv`: precision versus floating-point agreement, recall, false positives, and illustrative screening cost

Other feature counts use the same script: `--features 8` or `--features 32`. Each run replaces `reports/step4_quantization.csv` with that invocation only. Step 5 writes the full grid and then restores this file.

### RTL Simulation

```bash
python scripts/04_export_fixed_point.py --features 16 --bits 8 12 16 --simulate
```

The script compiles `hardware/linear_classifier.sv` and `hardware/classifier_tb.sv` once per bit width, with that width's generated header and vectors on the include path. The testbench checks **every validation vector** and returns a nonzero status on the first compile or simulate failure. A manual 16-bit run, from the generated directory (314 validation rows):

```bash
cd hardware/generated/k16_b16
iverilog -g2012 -s classifier_tb -DBITS=16 -DN=16 -DSAMPLES=314 -I . -o classifier_sim ../../linear_classifier.sv ../../classifier_tb.sv
vvp classifier_sim
```

Expect `PASS: 314 Python/RTL vector comparisons`. These vectors are validation rows, **not** the held-out test rows. Simulation checks functional agreement only. Do not publish area or timing from simulation. Step 5 is the area and timing run.

An early Windows `--simulate` attempt failed because `iverilog` was not on `PATH`. Do not claim RTL equivalence until this command prints `PASS` in the environment you are using. On a machine where Icarus Verilog 12.0 is installed, the 16-feature export matched all 314 vectors at 8, 12, and 16 bits.

### Quantization Results (16 features, 314 validation rows)

| Precision | Agreement with floating-point predictions | Failure recall | Illustrative screening cost |
| --- | ---: | ---: | ---: |
| 8-bit | 98.41% | 57.14% | 176 |
| 12-bit | **100.00%** | 57.14% | **175** |
| 16-bit | 99.68% | 57.14% | 176 |

Twelve-bit matching every floating-point decision, while 16-bit flips one boundary row, is a rounding effect. It is not a sign that more bits failed to synthesize.

### Sensitivity and Feature Stability

```bash
python experiments/validation_sensitivity.py
python experiments/feature_stability.py
```

Both use development rows only. Neither refits the saved classifier or scores the test set.

The chronological development split had **38.10% failure recall** at a fixed 0.50 threshold. That shows sensitivity to the split, not a deployment result. Five-fold failure recall on development data spanned about 37.5% to 58.8%.

Feature stability: `sensor_059`, `sensor_103`, and `sensor_510` were selected in all five folds.

## Step 5: Clocked wrapper and Yosys area/timing

`hardware/linear_classifier_top.sv` registers the feature word and the failure bit around the **unchanged** combinational classifier. Latency is **1 clock cycle**: inputs are captured on one rising edge, and the failure bit is captured on the next. `hardware/classifier_top_tb.sv` checks that registered bit against the same `vectors.mem` packing as Step 4.

```bash
conda install -c conda-forge yosys   # once, inside the course environment
python scripts/05_hardware_cost.py
```

Yosys and Icarus Verilog must be on `PATH`. The script, for every pair in {8, 12, 16} bits × {8, 16, 32} features:

1. Calls `scripts/04_export_fixed_point.py` for that one pair.
2. Simulates the clocked wrapper with Icarus.
3. Synthesizes the wrapper with Yosys.

The Nangate45 typical liberty file is downloaded on first use into `hardware/lib/` (gitignored). Scratch logs and netlists go to `hardware/synth/` (gitignored). Summaries are written to `reports/step5_hardware_cost.csv` and `reports/step5_hardware_cost.md`.

Area is the sum of Nangate45 cell areas (µm²). The critical path is the longest ABC path between flip-flops, plus DFF_X1 clock-to-Q and setup taken from the middle of those liberty tables. The ABC endpoint load is the DFF_X1 D-pin capacitance (1.14029 fF). Mapping uses ABC's direct `map` command after `synth -noabc`. Yosys's slower resizer was not used. A longer optimizer run could shrink area or delay somewhat. There is no wire-load model, no clock tree, and no power estimate. Do not describe these figures as silicon measurements. The script restores `reports/step4_quantization.csv` when it finishes, and it does not score the test set.

Expect nine lines of `PASS: 314 Python/RTL vector comparisons, latency_cycles=1`, then a line `Recommended: 16 features, 12 bits.` Under the published sweep that row matches every floating-point validation decision, has screening cost 175, about 10,847 cells, 193 flip-flops, about 11,628 µm², a critical path near 2.39 ns, and an estimated fmax near 418 MHz. The full grid is in `reports/step5_hardware_cost.md`.

## Tests

From the repository root, after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

The suite covers SECOM loading, the split, train-only scaling, the screening-cost helper, and fixed-point rounding, saturation, and the decision boundary. The development run reported **15 tests, all passing**.

What those tests and the RTL scripts do and do not cover:

- Python integer quantization, the decision boundary, clipping, and vector packing are covered by `unittest`.
- `scripts/04_export_fixed_point.py --simulate` marks RTL-versus-Python agreement `passed` only when Icarus exits successfully.
- Step 5 checks the clocked wrapper the same way and adds a Nangate45 area/timing estimate. That estimate is not an FPGA bitstream or a tapeout.
- No serial multiply-accumulate and no power number are claimed.
- The cost `10 × missed failures + 1 × false alarms` is illustrative only.
- The chronological experiment uses **development rows only**. It is a sensitivity check, not a prospective fab study.

## Results

Headline validation numbers. Full tables are in the step sections and in `reports/`.

- **Data.** 1,567 rows, 590 sensors, 104 failures (6.6%). Split 939 / 314 / 314 with 62 / 21 / 21 failures.
- **Always-pass baseline.** About 93.3% validation accuracy and no failures caught.
- **Software candidate for hardware.** 16-feature logistic regression, threshold 0.50: failure recall 57.1% (12 of 21), 85 false alarms, illustrative cost 175. A 32-feature random forest can score 172 under another scikit-learn version. The RTL exports the logistic model either way.
- **Quantization, 16 features.** 12-bit agrees with every floating-point validation decision (cost 175). 8-bit agrees on 98.41% (cost 176). 16-bit agrees on 99.68% (cost 176).
- **Stability.** Chronological development recall 38.10%. Sensors `sensor_059`, `sensor_103`, and `sensor_510` are selected in 5 of 5 folds.
- **Hardware.** 1-cycle latency at every width. Recommended estimate: 16 features, 12 bits, about 11,600 µm² and about 420 MHz on Nangate45 typical, with the caveats in [Step 5](#step-5-clocked-wrapper-and-yosys-areatiming).

## Optional Demo

```bash
python experiments/feature_stability.py
python -m pip install gradio
python dashboard/app.py
```

The Gradio demo loads **only validation sample indices** and does not expose test rows. Its slider changes the floating-point decision threshold. Displayed quantized decisions stay at the original trained threshold, and the page says so. This demo is not a production controller.

`experiments/feature_stability.py` counts how often selected sensors recur across five development-only training folds. `experiments/validation_sensitivity.py` adds the five-fold and chronological comparisons. Neither experiment updates the saved classifier or makes test-set claims.

## Caveats and Next Steps

- The hardware receives **already imputed, standardized, selected features**. Raw-sensor preprocessing is not in the RTL.
- All fitting and model comparison use train and validation rows. The held-out test partition has not been scored.
- The cost formula does **not** represent measured fab economics. Anonymous sensors mean a selected column is not a physical cause.
- Power is not estimated. The Nangate45 numbers are not a placed-and-routed design.
- Validation used 21 failures. Model and threshold choice on that same set can look better than a later test.
- Probability scores from class-weighted and oversampled models may be uncalibrated.
- RTL equivalence should be claimed only after `--simulate` or Step 5 prints `PASS` on the machine you are using.

**Next recommended work:** freeze the feature count, bit width, and threshold, then score the held-out test once. Do not tune after that score. Pin the scikit-learn version before quoting a final winner. A clocked or serial multiply-accumulate, a wire-load model, and a power estimate are separate designs from the current wrapper.
