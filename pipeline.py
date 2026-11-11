"""Shared preprocessing and scoring logic for the poverty targeting pipeline.

The preprocessing mirrors main.py (engineer_features) exactly, but stores everything it learns
(aggregated columns, imputation means, label encoders, column order) so new data can be scored.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

AGG_FUNCS = ['mean', 'min', 'max', 'sum']
GROUP_BINS = [0, 2, 5, 1000]
GROUP_LABELS = ['Small (1-2)', 'Medium (3-5)', 'Large (6+)']


def _aggregate(indiv: pd.DataFrame, num_cols):
    present = [c for c in num_cols if c in indiv.columns]
    if not present:
        return pd.DataFrame(index=indiv.index.unique())
    agg = indiv.groupby('id')[present].agg({c: AGG_FUNCS for c in present})
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg


class Preprocessor:
    """Learns the feature matrix layout on training data and reproduces it on new data."""

    def fit_transform(self, hhold: pd.DataFrame, indiv: pd.DataFrame) -> pd.DataFrame:
        hh_size = indiv.groupby('id').size().rename('hh_size')
        ic = indiv.drop(columns=[c for c in ['poor', 'iid'] if c in indiv.columns])
        self.num_cols = list(ic.select_dtypes(include=['int64', 'float64']).columns)
        agg = _aggregate(ic, self.num_cols)
        df = hhold.merge(agg, left_index=True, right_index=True, how='left')
        df = df.merge(hh_size, left_index=True, right_index=True, how='left')
        df.fillna(df.mean(numeric_only=True), inplace=True)
        df.fillna(-999, inplace=True)
        self.feature_cols = [c for c in df.columns if c != 'poor']
        self.hhold_cols = [c for c in hhold.columns if c != 'poor']
        self.means = df[self.feature_cols].apply(pd.to_numeric, errors='coerce').mean()
        self.obj_cols = list(df[self.feature_cols].select_dtypes(include=['object']).columns)
        self.enc_maps = {}
        for col in self.obj_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.enc_maps[col] = {cls: i for i, cls in enumerate(le.classes_)}
        return df

    def transform(self, hhold: pd.DataFrame, indiv: pd.DataFrame):
        """Return (X, report) where report lists data-quality warnings."""
        report = {}
        hh_size = indiv.groupby('id').size().rename('hh_size')
        ic = indiv.drop(columns=[c for c in ['poor', 'iid'] if c in indiv.columns])
        agg = _aggregate(ic, self.num_cols)
        base = hhold.drop(columns=['poor'], errors='ignore')
        df = base.merge(agg, left_index=True, right_index=True, how='left')
        df = df.merge(hh_size, left_index=True, right_index=True, how='left')
        missing = [c for c in self.feature_cols if c not in df.columns]
        report['missing_columns'] = len(missing)
        report['expected_columns'] = len(self.feature_cols)
        report['households_without_individuals'] = int(df['hh_size'].isna().sum()) if 'hh_size' in df else len(df)
        df = df.reindex(columns=self.feature_cols)
        for col in self.feature_cols:
            if col not in self.obj_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(self.means).fillna(-999)
        for col in self.obj_cols:
            df[col] = df[col].astype(str).map(self.enc_maps[col]).fillna(-1).astype(int)
        return df.astype(float), report


def targeting_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics for one model. Returns None for values that are undefined on this data."""
    out = {}
    both = len(np.unique(y)) == 2
    out['log_loss'] = float(log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1]))
    out['roc_auc'] = float(roc_auc_score(y, prob)) if both else None
    out['accuracy'] = float(accuracy_score(y, pred))
    out['f1'] = float(f1_score(y, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out['exclusion_error'] = float(fn / (fn + tp)) if (fn + tp) else None
    out['inclusion_error'] = float(fp / (fp + tn)) if (fp + tn) else None
    return out


def subgroup_table(hh_size: pd.Series, y: np.ndarray, pred_lgb: np.ndarray, pred_pmt: np.ndarray):
    groups = pd.cut(hh_size.values, bins=GROUP_BINS, labels=GROUP_LABELS)
    rows = []
    for label in GROUP_LABELS:
        m = np.asarray(groups == label)
        poor = int(y[m].sum())
        def excl(p):
            if poor == 0:
                return None
            return float(((y[m] == 1) & (p[m] == 0)).sum() / poor)
        rows.append({'group': label, 'households': int(m.sum()), 'poor': poor,
                     'exclusion_lgb': excl(pred_lgb), 'exclusion_pmt': excl(pred_pmt)})
    return rows
