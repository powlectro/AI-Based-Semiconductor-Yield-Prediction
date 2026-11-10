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

## AI assistance disclosure
The Georgia Tech course syllabus permits AI coding assistants as a learning
tool with disclosure. Follow your instructor's/team's disclosure expectations.
