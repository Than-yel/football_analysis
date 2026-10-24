# Premier League Match Outcome Prediction — Machine Learning Study

Experimental three-class (home win / draw / away win) Premier League classification project, using the provided historical match records through **19 October 2026**.

## Data and methodology

- `data/EPL_data.csv`: 1,590 rows (four full seasons, 2022/23–2025/26, plus 70 2026/27 matches), 26 fields, assembled from the supplied season files. Validated 0 missing values, 0 duplicate season fixtures and 27 teams.
- Predictions use historical-only form, venue and season summaries, rest, Elo, and history-count features; never same-match statistics.
- Expanding-season validation selects model settings on 2023/24–2025/26; 2026/27's 70 matches are the final holdout.
- Models: multinomial logistic regression, random forest and XGBoost. Random Forest was selected by mean chronological-validation **log loss**, not test accuracy.

## Results

| Model | Final test accuracy | Final test log loss |
|---|---:|---:|
| Class-prior baseline | 40.0% | 1.095 |
| Logistic Regression | 44.3% | 1.151 |
| Random Forest (selected) | 44.3% | 1.022 |
| XGBoost | 45.7% (original saved run) | 1.028 (original saved run) |

The final holdout includes only 70 matches. The champion's **draw recall is 0%**; accuracy is modest. These are experimental probabilities, not dependable guarantees. During independent verification using a different package environment, XGBoost obtained 44.3% accuracy and 1.023 log loss, while the Random Forest and its forecasts reproduced to the displayed precision. This discrepancy demonstrates the importance of the recorded dependency versions.

## Weekend forecast and cutoff

`predictions/matchweek8_champion_predictions.csv` contains the nine Saturday/Sunday forecasts for 24–25 October 2026 (home/draw/away probabilities). The notebook separately labels Friday's Ipswich–Nottingham Forest game as a **backcast**, not a live prediction. Models use only completed matches through 19 October. Forecast fixture kickoff timestamps are stored in UK local time with WAT display times.

## Running the notebook

From the project root:

```bash
python -m venv .venv
# Activate the environment for your OS
pip install -r requirements.txt
jupyter notebook notebooks/epl_match_outcome_prediction.ipynb
```

The notebook checks relative data paths, imports `src/epl_feature_builder.py` and writes generated outputs under `artifacts/` (ignored by Git). Notebook outputs in the committed copy are **saved results from the original run**, not the newer independent verification. `src/epl_feature_builder.py` was reconstructed from the notebook's own historical feature-building code because the original external module was not supplied.

## Verification

The independent complete notebook execution verified the input cutoff and target contract, chronological train/test separation, 93-feature generation, Random Forest accuracy/log loss, nine probabilities, and save/reload parity (maximum difference under 2e-16). Source match provenance was not independently checked against official records. Historical team priors and promoted-team assumptions remain limitations. This is a research prototype, **not production-ready**.
