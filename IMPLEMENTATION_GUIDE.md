# Stages 4–6: fixed-point and SystemVerilog

This extends the **existing** Steps 1–3 rather than replacing them. **Do not evaluate the held-out test set until feature count, precision and threshold are frozen.**

## Setup (repository root)

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Train and export

```bash
python scripts/03_train_compare.py
python scripts/04_export_fixed_point.py --features 16 --bits 8 12 16
python experiments/validation_sensitivity.py
```

`04_export_fixed_point.py` uses the **saved train/validation split** and trained `artifacts/step3_logistic_k16.joblib`. It does not fit another model or inspect the test partition. It produces:

- `hardware/generated/k16_b08/`, `k16_b12/`, `k16_b16/`
- `model_constants.svh`: signed quantized coefficients and comparison offset
- `vectors.mem`: *validation* test vectors and bit-accurate expected predictions (expected bit is MSB; sensor zero occupies the LSB)
- `config.json`: input/weight fractional bits, ordered sensor names, and threshold
- `reports/step4_quantization.csv`: precision vs floating-point agreement, recall, false positives, and illustrative screening cost

The **Python bit-accurate reference** is `src/fixed_point.py`. Inputs are train-median-imputed, standardized, selected sensors, **not raw wafer measurements**. The preprocessor and feature order remain in the trusted local `joblib` artifacts. The hardware input interface assumes preprocessing happened upstream.

## Run RTL simulation (Icarus Verilog)

Install `iverilog` and `vvp` separately and ensure both are on your PATH. Then:

```bash
python scripts/04_export_fixed_point.py --features 16 --bits 8 12 16 --simulate
```

The script compiles `hardware/linear_classifier.sv` + `hardware/classifier_tb.sv` separately for each bit width, using the generated constant header and vectors. The testbench checks **every validation vector**, and returns a nonzero exit status on the first compilation/simulation failure. Example manual run for 16-bit:

```bash
cd hardware/generated/k16_b16
iverilog -g2012 -s classifier_tb -DBITS=16 -DN=16 -DSAMPLES=314 -I . -o classifier_sim ../../linear_classifier.sv ../../classifier_tb.sv
vvp classifier_sim
```

The generated `vectors.mem` uses *validation* rows (currently 314), **not held-out test rows**. Do not publish timing/area claims from simulation alone. For synthesis, use a target FPGA/ASIC flow with a technology library and report LUTs/DSPs or cell area, plus critical path and clock constraints. The provided combinational architecture has no registers or clock, so its timing is a combinational path, not a measured clock-cycle latency. A pipelined/serial MAC would be a separate implementation.

## What is and isn't verified

- Python integer quantization, decision boundary, clipping, and packing are tested with `unittest`.
- Run `--simulate` with Icarus to verify actual RTL-vs-Python prediction agreement; the script marks the status `passed` only if simulator succeeds.
- No clocked pipeline, serial MAC, physical area/timing, or FPGA deployment is claimed.
- The numerical cost (10 × missed failures + 1 × false alarms) is illustrative only.
- The chronological experiment is based on **development rows only** and is a sensitivity analysis, not a substitute for a later prospective fab study.

## Add to GitHub

See the response accompanying this ZIP for `git` commands. Generated RTL vectors and trained `.joblib` files are intentionally gitignored; commit source code and reports, not machine-specific binaries.

## Optional demos and analysis

```bash
python experiments/feature_stability.py
python -m pip install gradio
python dashboard/app.py
```

The Gradio demo loads ONLY validation sample indices and does not expose test rows. Its slider changes the *float* decision threshold; displayed quantized decisions remain at the original trained threshold, which is explicitly labeled. This demo is not a semiconductor production controller.

`experiments/feature_stability.py` computes how often selected sensors recur across five development-only training folds. `experiments/validation_sensitivity.py` computes a separate repeated-fold (five-fold) and chronological sensitivity comparison. Neither experiment updates the deployed classifier or makes test-set claims.

### Important environment note

The repository's dependencies use open lower bounds, so rerunning Step 3 under a newer `scikit-learn` version may change fitted coefficients and selected validation thresholds. Record package versions alongside published results, and freeze a requirements lockfile for a final evaluation. Do **not** silently compare figures from runs under different library versions.
