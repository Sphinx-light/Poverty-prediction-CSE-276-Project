# Household Poverty Prediction for Programme Targeting

CSE 276 capstone (Lovely Professional University). Predicts which households are poor using the DrivenData **Pover-T** survey data (countries A, B, C) and compares a LightGBM model with a Proxy Means Test (PMT) logistic regression baseline. A small FastAPI service lets anyone upload their own household data and get scores back.

Live site: https://sphinx-light.github.io/poverty_prediction_site/ (source in the `poverty_prediction_site` repository).

## What is in this repository

| File / folder | Purpose |
|---|---|
| `main.py` | Full research script: BFS route planning, data loading, feature engineering, PMT baseline, LightGBM, subgroup analysis, plots |
| `app.py` | FastAPI service (`/health`, `/meta`, `/predict`) |
| `pipeline.py` | Shared preprocessing and metric code used by training and the API |
| `train.py` | Trains the final models and writes `artifacts/{A,B,C}.joblib` |
| `make_sample.py` | Builds small sample upload files from the training CSVs |
| `artifacts/` | Trained models for countries A, B and C |
| `requirements.txt` | Python dependencies |

## Method in brief

- **Features:** numeric individual-level columns are aggregated per household (mean, min, max, sum) plus household size, then joined to household columns. Missing values are filled with the global mean (then -999), and categoricals are label-encoded.
- **LightGBM:** binary objective, learning rate 0.05, max depth 6, 31 leaves, feature fraction 0.8, class weighting, 5-fold stratified cross-validation (seed 42), early stopping at 50 rounds.
- **PMT baseline:** logistic regression on standardised features, 0.5 cutoff.
- **Targeting metrics:** the poverty-rate threshold is set from the out-of-fold prediction percentile. Reported are log loss, ROC-AUC, exclusion error (poor households missed), inclusion error (non-poor households wrongly included) and household-size subgroup results.
- **BFS section:** a breadth-first search over an intervention graph to plan a resource-allocation route.

## 1. Reproduce the study

Put the six Pover-T training CSVs (`A_hhold_train.csv`, `A_indiv_train.csv`, and the same for B and C) next to `main.py`. They come from the DrivenData competition and are not redistributed here; download them from the competition page.

```bash
pip install -r requirements.txt
pip install matplotlib seaborn
python main.py
```

It prints the BFS result and the per-country PMT and LightGBM metrics, and saves the feature-importance and targeting-dashboard PNGs.

## 2. Run the API locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Check it at http://localhost:8000/health (returns `{"status": "ok"}`).

### Endpoints

- `GET /health` – liveness check.
- `GET /meta` – available countries and model details.
- `POST /predict` – multipart form with `country` (A, B or C), `hhold` (CSV), `indiv` (CSV), optional `threshold_mode` (`saved` or `rate`) and `poverty_rate`. If the household file contains a `poor` column, accuracy metrics and subgroup tables are returned too. Returns a summary, warnings and up to 20,000 predictions.

### Try it with a sample

```bash
python make_sample.py --data-dir path/to/pover-t-csvs --country C --n 300
```

This writes `sample_C_hhold.csv` and `sample_C_indiv.csv`. They come from the training data, so scores look better than they would on new households.

## 3. Retrain the models

```bash
python train.py --data-dir path/to/pover-t-csvs --out artifacts
```

Cross-validation matches `main.py`, so the printed metrics match the report.

## 4. Deploy on Render (free tier)

1. Push this repository to GitHub.
2. On render.com choose **New > Web Service** and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Environment variable `ALLOWED_ORIGINS` = `https://sphinx-light.github.io` (no path, no trailing slash; use `*` only for testing). Several origins can be separated by commas.
6. Copy the Render URL into the "Model API address" box on the website.

Free instances sleep when idle, so the first request can take about a minute.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Websites allowed to call the API (CORS) |
| `MAX_UPLOAD_MB` | `40` | Maximum upload size |

## Notes and limitations

- Uploads are processed in memory and never written to disk.
- Each country model accepts only its own columns; files missing more than half of them are rejected.
- Final models are fitted on all training rows, so metrics on training households are optimistic.
- Probabilities come from a class-weighted model. Use them for ranking, not as exact risk.
- Data and trained models derive from the DrivenData Pover-T competition. Check its terms before reusing them.
