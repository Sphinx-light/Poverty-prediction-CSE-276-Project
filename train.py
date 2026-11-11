"""Train the PMT baseline and LightGBM targeting model for each country and save them for the API.

Usage:  python train.py --data-dir path/to/csvs --out artifacts
Expects {A,B,C}_hhold_train.csv and {A,B,C}_indiv_train.csv in the data directory.
Cross-validation settings match main.py, so thresholds and metrics are the same as the report.
"""
import argparse
import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pipeline import Preprocessor

warnings.filterwarnings('ignore')
N_SPLITS, SEED = 5, 42
LGBM_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05,
               'max_depth': 6, 'num_leaves': 31, 'feature_fraction': 0.8, 'verbose': -1}


def train_country(country, data_dir, out_dir):
    hh = pd.read_csv(os.path.join(data_dir, f'{country}_hhold_train.csv'), index_col='id')
    iv = pd.read_csv(os.path.join(data_dir, f'{country}_indiv_train.csv'), index_col='id')
    pre = Preprocessor()
    df = pre.fit_transform(hh, iv)
    X = df[pre.feature_cols]
    y = df['poor'].astype(int)
    params = {**LGBM_PARAMS, 'scale_pos_weight': (y == 0).sum() / max((y == 1).sum(), 1)}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof, best_iters, imp = np.zeros(len(X)), [], np.zeros(X.shape[1])
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        imp += m.feature_importance('split') / N_SPLITS

    rate = float(y.mean())
    threshold = float(np.percentile(oof, (1 - rate) * 100))
    pred = (oof >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rounds = int(round(np.mean(best_iters)))
    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)

    scaler = StandardScaler().fit(X)
    pmt = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(X), y)

    top = pd.Series(imp, index=X.columns).sort_values(ascending=False).head(5)
    meta = {'country': country, 'households': int(len(y)), 'poor_households': int(y.sum()),
            'poverty_rate': rate, 'threshold': threshold, 'boost_rounds': rounds,
            'n_features': int(X.shape[1]),
            'oof_log_loss': float(log_loss(y, oof)), 'oof_roc_auc': float(roc_auc_score(y, oof)),
            'oof_exclusion_error': float(fn / (fn + tp)), 'oof_inclusion_error': float(fp / (fp + tn)),
            'top_features': {k: round(float(v), 1) for k, v in top.items()}}
    joblib.dump({'pre': pre, 'lgb_model': final.model_to_string(), 'scaler': scaler, 'pmt': pmt, 'meta': meta},
                os.path.join(out_dir, f'{country}.joblib'), compress=3)
    print(f"{country}: logloss {meta['oof_log_loss']:.4f}  auc {meta['oof_roc_auc']:.4f}  "
          f"excl {meta['oof_exclusion_error']:.4f}  threshold {threshold:.4f}  rounds {rounds}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--out', default='artifacts')
    ap.add_argument('--countries', default='A,B,C')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.countries.split(','):
        train_country(c.strip(), a.data_dir, a.out)
