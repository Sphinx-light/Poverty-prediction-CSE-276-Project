"""FastAPI service that scores uploaded household and individual CSVs.

Run locally:   uvicorn app:app --reload
Environment:   ARTIFACT_DIR (default ./artifacts), ALLOWED_ORIGINS (comma list, default *), MAX_UPLOAD_MB (default 40)
Uploaded files are processed in memory and never written to disk.
"""
import io
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import subgroup_table, targeting_metrics

ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', 'artifacts')
MAX_BYTES = int(float(os.environ.get('MAX_UPLOAD_MB', '40')) * 1024 * 1024)
MAX_ROWS_RETURNED = 20000
ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]

app = FastAPI(title='Household poverty targeting API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST'], allow_headers=['*'])

MODELS = {}


@app.on_event('startup')
def load_models():
    for c in 'ABC':
        path = os.path.join(ARTIFACT_DIR, f'{c}.joblib')
        if os.path.exists(path):
            bundle = joblib.load(path)
            bundle['booster'] = lgb.Booster(model_str=bundle.pop('lgb_model'))
            MODELS[c] = bundle


@app.get('/health')
def health():
    return {'status': 'ok', 'countries': sorted(MODELS)}


@app.get('/meta')
def meta():
    return {c: b['meta'] for c, b in MODELS.items()}


def _read_csv(upload: UploadFile, what: str) -> pd.DataFrame:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f'{what} file is larger than {MAX_BYTES // (1024 * 1024)} MB.')
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, f'Could not read the {what} file as CSV.')
    if 'id' not in df.columns:
        raise HTTPException(400, f"The {what} file needs an 'id' column (household id).")
    return df.set_index('id')


def _num(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


@app.post('/predict')
def predict(country: str = Form(...), hhold: UploadFile = File(...), indiv: UploadFile = File(...),
            threshold_mode: str = Form('saved'), poverty_rate: float = Form(None)):
    country = country.upper()
    if country not in MODELS:
        raise HTTPException(404, f'No model loaded for country {country}. Available: {sorted(MODELS)}')
    if threshold_mode not in ('saved', 'rate'):
        raise HTTPException(400, "threshold_mode must be 'saved' or 'rate'.")
    b = MODELS[country]
    hh, iv = _read_csv(hhold, 'household'), _read_csv(indiv, 'individual')
    if len(hh) == 0 or len(iv) == 0:
        raise HTTPException(400, 'Both files need at least one row.')
    if hh.index.has_duplicates:
        raise HTTPException(400, 'The household file has duplicate ids. Use one row per household.')

    X, report = b['pre'].transform(hh, iv)
    warnings = []
    if report['missing_columns'] > 0.5 * report['expected_columns']:
        raise HTTPException(400, f"This does not look like a Country {country} file: "
                                 f"{report['missing_columns']} of {report['expected_columns']} expected columns are missing.")
    if report['missing_columns']:
        warnings.append(f"{report['missing_columns']} of {report['expected_columns']} expected columns are missing and were filled with training averages.")
    if report['households_without_individuals']:
        warnings.append(f"{report['households_without_individuals']} households have no individual rows, so their person-level features are filled with training averages.")

    prob = b['booster'].predict(X)
    pmt_prob = b['pmt'].predict_proba(b['scaler'].transform(X))[:, 1]

    expected_rate = float(poverty_rate) if poverty_rate is not None else b['meta']['poverty_rate']
    if not 0 < expected_rate < 1:
        raise HTTPException(400, 'poverty_rate must be between 0 and 1.')
    threshold, mode = b['meta']['threshold'], 'saved'
    if threshold_mode == 'rate':
        if len(X) >= 50:
            threshold, mode = float(np.percentile(prob, (1 - expected_rate) * 100)), 'rate'
        else:
            warnings.append('Fewer than 50 households, so the saved training threshold was used instead of a poverty-rate threshold.')
    pred = (prob >= threshold).astype(int)
    pmt_pred = (pmt_prob > 0.5).astype(int)

    result = {'country': country, 'n_households': int(len(X)), 'threshold_mode': mode,
              'threshold': threshold, 'expected_poverty_rate': expected_rate,
              'predicted_poor': int(pred.sum()), 'predicted_poor_share': float(pred.mean()),
              'warnings': warnings, 'has_labels': False, 'metrics': None, 'subgroups': None,
              'note': 'Probabilities come from a class-weighted model, so rank households by probability instead of reading them as exact risk.'}

    if 'poor' in hh.columns:
        y = hh.loc[X.index, 'poor'].astype(str).str.lower().map({'true': 1, '1': 1, '1.0': 1, 'false': 0, '0': 0, '0.0': 0})
        if y.notna().all():
            y = y.astype(int).values
            result['has_labels'] = True
            result['metrics'] = {'lightgbm': {k: _num(v) for k, v in targeting_metrics(y, prob, pred).items()},
                                 'pmt': {k: _num(v) for k, v in targeting_metrics(y, pmt_prob, pmt_pred).items()}}
            result['subgroups'] = subgroup_table(X['hh_size'], y, pred, pmt_pred)
        else:
            warnings.append("The 'poor' column has values that are not True/False or 1/0, so accuracy metrics were skipped.")

    rows = [{'id': str(i), 'probability': float(p), 'predicted_poor': bool(d), 'pmt_probability': float(q)}
            for i, p, d, q in zip(X.index[:MAX_ROWS_RETURNED], prob, pred, pmt_prob)]
    result['predictions'] = rows
    result['truncated'] = len(X) > MAX_ROWS_RETURNED
    return result
