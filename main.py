"""
Household Poverty Prediction & Resource Allocation System
=========================================================
Pipeline: BFS Route Planning → Data Loading → Feature Engineering 
          → PMT Baseline (Logistic Regression) → LightGBM Targeting 
          → Subgroup Analysis → Dashboard Visualisation

Supports Countries A, B, C (DrivenData Pover-T competition format).
"""

import os
import time
from collections import deque
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of this script
N_SPLITS = 5
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")

LGBM_BASE_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'learning_rate':     0.05,
    'max_depth':         6,
    'num_leaves':        31,
    'feature_fraction':  0.8,
    'verbose':           -1,
}


# ===========================================================================
# SECTION 1 — STATE-SPACE SEARCH ROUTE PLANNING (BFS)
# ===========================================================================

def bfs_poverty_planner(graph, start, goal):
    """
    Breadth First Search for finding the minimum-hop intervention route.
    Follows the exact pattern from Practical 1.
    """
    queue = deque([[start]])
    visited = set()
    explored = []

    start_time = time.perf_counter()

    while queue:
        path = queue.popleft()          # FIFO queue operation
        current = path[-1]

        if current in visited:
            continue

        visited.add(current)
        explored.append(current)

        if current == goal:
            elapsed_time = time.perf_counter() - start_time
            return path, explored, elapsed_time

        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                queue.append(path + [neighbour])

    elapsed_time = time.perf_counter() - start_time
    return None, explored, elapsed_time


# ===========================================================================
# SECTION 2 — DATA LOADING & FEATURE ENGINEERING (FAST VECTORIZED)
# ===========================================================================

def load_data(country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load household and individual CSVs for a given country code."""
    hhold = pd.read_csv(os.path.join(DATA_DIR, f"{country}_hhold_train.csv"), index_col='id')
    indiv = pd.read_csv(os.path.join(DATA_DIR, f"{country}_indiv_train.csv"), index_col='id')
    return hhold, indiv


def engineer_features(hhold: pd.DataFrame, indiv: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized feature engineering and aggregation."""
    # Calculate household size from individual records
    hh_size = indiv.groupby('id').size().rename('hh_size')
    
    # Select numeric columns for fast aggregation, excluding target/ID columns
    drop_cols = [c for c in ['poor', 'iid'] if c in indiv.columns]
    indiv_clean = indiv.drop(columns=drop_cols)
    
    num_cols = indiv_clean.select_dtypes(include=['int64', 'float64']).columns
    agg_funcs = {col: ['mean', 'min', 'max', 'sum'] for col in num_cols}
    
    # Fast C-optimized pandas groupby aggregation
    indiv_agg = indiv_clean.groupby('id').agg(agg_funcs)
    indiv_agg.columns = ['_'.join(col).strip() for col in indiv_agg.columns.values]
    
    # Merge with household table
    df = hhold.merge(indiv_agg, left_index=True, right_index=True, how='left')
    df = df.merge(hh_size, left_index=True, right_index=True, how='left')
    
    # Impute missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(-999, inplace=True)
    
    # Encode categorical variables efficiently
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df


# ===========================================================================
# SECTION 3 — MODELS & TARGETING
# ===========================================================================

def run_pmt_baseline(X: pd.DataFrame, y: pd.Series) -> dict:
    """Proxy Means Test baseline: stratified-CV Logistic Regression."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X)

    for tr_idx, va_idx in skf.split(X_arr, y):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_arr[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X_arr[va_idx])[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, (oof_preds > 0.5).astype(int)).ravel()
    return {
        'LogLoss':        log_loss(y, oof_preds),
        'ROC-AUC':        roc_auc_score(y, oof_preds),
        'ExclusionError': fn / (fn + tp) if (fn + tp) > 0 else 0,
    }


def run_lightgbm(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """LightGBM targeting model with stratified CV and class-weight balancing."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    feat_imp  = np.zeros(X.shape[1])

    # Compensate for class imbalance automatically
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    params = {**LGBM_BASE_PARAMS, 'scale_pos_weight': pos_weight}

    for tr_idx, va_idx in skf.split(X, y):
        train_ds = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        val_ds   = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=train_ds)

        model = lgb.train(
            params, train_ds,
            num_boost_round=1000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_preds[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        feat_imp += model.feature_importance('split') / N_SPLITS

    fi_df = (
        pd.DataFrame({'Feature': X.columns, 'Importance': feat_imp})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )
    return oof_preds, fi_df


# ===========================================================================
# SECTION 4 — ANALYSIS & VISUALISATION
# ===========================================================================

def subgroup_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exclusion error by household-size subgroup."""
    rows = []
    for group, sub in df.groupby('hh_size_group', observed=True):
        fn = ((sub['actual_poor'] == 1) & (sub['pred_poor'] == 0)).sum()
        tp = ((sub['actual_poor'] == 1) & (sub['pred_poor'] == 1)).sum()
        rows.append({
            'Subgroup':            group,
            'Exclusion Error (%)': fn / (fn + tp) * 100 if (fn + tp) > 0 else 0,
            'Total Poor':           fn + tp,
        })
    return pd.DataFrame(rows)


def plot_feature_importance(fi_df: pd.DataFrame, country: str, out_dir: str):
    top10 = fi_df.head(10).sort_values('Importance')
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top10)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10['Feature'], top10['Importance'], color=colors)
    ax.set_title(f'Top 10 Feature Importances — Country {country}')
    ax.set_xlabel('Importance (split count)')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    path = os.path.join(out_dir, f'feature_importance_{country}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dashboard(subgroup_df: pd.DataFrame, pmt: dict, lgb_auc: float,
                   lgb_excl: float, country: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: subgroup exclusion errors
    sns.barplot(x='Subgroup', y='Exclusion Error (%)',
                data=subgroup_df, ax=axes[0], palette='Reds_d')
    axes[0].set_title(f'Country {country}: Exclusion Error by Household Size')
    axes[0].set_ylabel('Exclusion Error (%) [Poor Wrongly Excluded]')
    axes[0].set_ylim(bottom=0)

    # Right: PMT vs LightGBM metric comparison
    comp = pd.DataFrame({
        'Model':                ['PMT Baseline', 'LightGBM Classifier'],
        'ROC-AUC':              [pmt['ROC-AUC'],        lgb_auc],
        'Exclusion Error Rate': [pmt['ExclusionError'], lgb_excl],
    }).melt(id_vars='Model', var_name='Metric', value_name='Value')
    sns.barplot(x='Metric', y='Value', hue='Model', data=comp, ax=axes[1], palette='Set2')
    axes[1].set_title(f'Country {country}: PMT Baseline vs LightGBM')

    fig.tight_layout()
    path = os.path.join(out_dir, f'targeting_dashboard_{country}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 5 — COUNTRY PIPELINE EXECUTION
# ===========================================================================

def process_country(country: str):
    print(f"\n{'='*50}")
    print(f"  TARGETING ANALYSIS: COUNTRY {country}")
    print(f"{'='*50}")

    try:
        hhold, indiv = load_data(country)
    except FileNotFoundError:
        print(f"  [SKIP] Data files for Country {country} not found.")
        return

    df = engineer_features(hhold, indiv)
    X  = df.drop(columns=['poor'])
    y  = df['poor'].astype(int)
    print(f"  Dataset: {len(y)} households | Poverty rate: {y.mean():.1%}")

    # Models
    pmt = run_pmt_baseline(X, y)
    oof_pred, fi_df = run_lightgbm(X, y)

    # Dynamic thresholding based on actual headcount poverty rate
    threshold   = np.percentile(oof_pred, (1 - y.mean()) * 100)
    binary_pred = (oof_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, binary_pred).ravel()
    lgb_metrics = {
        'LogLoss':        log_loss(y, oof_pred),
        'ROC-AUC':        roc_auc_score(y, oof_pred),
        'ExclusionError': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'InclusionError': fp / (fp + tn) if (fp + tn) > 0 else 0,
    }

    # Print summary comparison
    print(f"\n  {'Metric':<26} {'PMT Baseline':>14} {'LightGBM':>12}")
    print(f"  {'-'*54}")
    for key in ('LogLoss', 'ROC-AUC', 'ExclusionError'):
        print(f"  {key:<26} {pmt[key]:>14.4f} {lgb_metrics[key]:>12.4f}")
    print(f"  {'Threshold (LightGBM)':26} {threshold:>26.4f}")
    print(f"  {'Inclusion Error (LGB)':26} {lgb_metrics['InclusionError']:>26.4f}")

    # Subgroup analysis by household size
    df['pred_poor']   = binary_pred
    df['actual_poor'] = y
    df['hh_size_group'] = pd.cut(
        df['hh_size'], bins=[0, 2, 5, 20],
        labels=['Small (1-2)', 'Medium (3-5)', 'Large (6+)']
    )
    sub_df = subgroup_exclusion(df)
    print(f"\n  Subgroup Exclusion Error Breakdown:")
    print(sub_df.to_string(index=False))

    # Visualisations
    out_dir = DATA_DIR
    plot_feature_importance(fi_df, country, out_dir)
    plot_dashboard(sub_df, pmt, lgb_metrics['ROC-AUC'],
                   lgb_metrics['ExclusionError'], country, out_dir)


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    # --- Part A: State-Space Search Route Planning ---
    print(f"\n{'='*50}")
    print("   RESOURCE ALLOCATION ROUTE PLANNING (BFS)")
    print(f"{'='*50}")
    
    intervention_graph = {
        'HQ': ['District_A', 'District_B'],
        'District_A': ['Community_1', 'Community_2'],
        'District_B': ['Community_3'],
        'Community_1': ['Target_Zone_Alpha'],
        'Community_2': ['Target_Zone_Beta'],
        'Community_3': ['Target_Zone_Alpha'],
        'Target_Zone_Alpha': [],
        'Target_Zone_Beta': []
    }

    start_node = 'HQ'
    goal_node = 'Target_Zone_Alpha'

    path, explored, runtime = bfs_poverty_planner(intervention_graph, start_node, goal_node)
    print(f"Optimal Hop Path     : {' -> '.join(path) if path else 'No Path Found'}")
    print(f"Explored Nodes ({len(explored)}) : {explored}")
    print(f"Execution Time       : {runtime:.6f} seconds")

    # --- Part B: Machine Learning Targeting Analysis & Plotting ---
    for country in ['A', 'B', 'C']:
        process_country(country)

    print("\nAll pipeline tasks and visualizations completed successfully.")