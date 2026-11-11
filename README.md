# Poverty targeting API

Scores uploaded household + individual CSVs with the LightGBM and PMT models from the CSE 276 project.
The website (`poverty.html`) talks to this service. The trained models for countries A, B and C are in `artifacts/`.

## Run it on your computer
```
pip install -r requirements.txt
uvicorn app:app --port 8000
```
Open `poverty.html`, scroll to "Try it", keep the address `http://localhost:8000`, and press "Check connection".

## Try it with a sample
```
python make_sample.py --data-dir path/to/pover-t-csvs --country C --n 300
```
This writes `sample_C_hhold.csv` and `sample_C_indiv.csv`. They come from the training data, so scores look unrealistically good.

## Put it online (Render, free tier)
1. Push this folder to a GitHub repository.
2. On render.com choose New > Web Service and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add the environment variable `ALLOWED_ORIGINS` set to the address where your website is hosted (for example `https://yourname.github.io`). Without it any site can call your API.
6. Copy the Render address into the "Model API address" box on the website, or set `DEFAULT_API_URL` in `poverty.html`.

The website must be on https if the API is on https. Free instances sleep when idle, so the first request can take about a minute.

## Retrain
```
python train.py --data-dir path/to/pover-t-csvs --out artifacts
```
Cross-validation matches `main.py` (5 folds, seed 42), so the printed log loss, AUC, exclusion error and thresholds match the report.

## Notes
- Uploads are read in memory and never written to disk. Limit: `MAX_UPLOAD_MB` (default 40).
- Each country model only accepts its own columns. Files that miss more than half of them are rejected.
- Final models are fitted on all training rows. Metrics on households from the training data are optimistic.
- Probabilities come from a class-weighted model. Use them for ranking, not as exact risk.
- The trained models and any sample files come from the DrivenData Pover-T data. Check the competition terms before publishing them.
