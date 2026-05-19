# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches mpatches
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ===================================================================
# Global Configuration
# ===================================================================
FILE = 'FGAS_filled.xlsx'
GAS = 'FGAS'
OUT_DIR = f'output_SHAP_{GAS}'
N_TRIALS = 50

# ===================================================================
# Dimension Color and Feature Mapping
# ===================================================================
DIM_COLORS = {
    'Economy': '#2196F3',
    'Society': '#4CAF50',
    'Industry': '#FF9800',
    'Technology': '#9C27B0',
    'Policy': '#F44336',
    'Environment': '#00BCD4',
}

FEATURE_DIM = {
    'GDP': 'Economy', 'PGDP': 'Economy',
    'CCI': 'Economy', 'OAGCI': 'Economy',
    'POP': 'Society', 'URB': 'Society', 'DS': 'Society',
    'IVA': 'Industry', 'IOVP': 'Industry',
    'INDS': 'Technology',
    'INDP': 'Technology',
    'POL': 'Policy',
    'TEMP': 'Environment', 'PRCP': 'Environment',
}

# ===================================================================
# Step 1: Data Preparation and Feature Engineering
# ===================================================================
def prepare_data(file_path):
    print('=' * 60)
    print('[Step 1] Loading Data and Constructing Features')

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values('Year').reset_index(drop=True)
    print(f'  Original columns: {list(df.columns)}')

    col_map = {
        '气温': 'TEMP',
        '降水': 'PRCP',
        '人口': 'POP',
        'GDP': 'GDP',
        '人均GDP': 'PGDP',
        '城镇化率': 'URB',
        '饮食结构': 'DS',
        '政策数量': 'POL',
        '工业产值': 'IVA',
        'Industrial Processes_技术规模': 'INDS',
        'Industrial Processes_技术渗透率': 'INDP',
    }
    df = df.rename(columns=col_map)

    eps = 1e-9

    if '煤炭消费量' in df.columns:
        df['CCI'] = df['煤炭消费量'] / (df['GDP'] + eps)
        print('  -> CCI constructed successfully')

    oil_cols = [c for c in ['石油消费量', '天然气消费量'] if c in df.columns]
    if oil_cols:
        df['OAGCI'] = df[oil_cols].sum(axis=1) / (df['GDP'] + eps)
        print(f'  -> OAGCI constructed successfully (Based on {oil_cols})')

    if 'IVA' in df.columns:
        df['IOVP'] = df['IVA'] / (df['GDP'] + eps)
        print('  -> IOVP constructed successfully')

    fgas_features = [
        'GDP', 'PGDP', 'CCI', 'OAGCI',
        'POP', 'URB', 'DS',
        'IVA', 'IOVP',
        'INDS', 'INDP',
        'POL',
        'TEMP', 'PRCP',
    ]

    feature_cols = [c for c in fgas_features if c in df.columns]
    missing = set(fgas_features) - set(feature_cols)
    if missing:
        print(f'  Warning: Features unavailable: {sorted(missing)}')
    print(f'  Selected Features ({len(feature_cols)}): {feature_cols}')

    X = df[feature_cols].copy()
    y = df['排放量'].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        print(f'  Removing constant columns: {constant_cols}')
        X = X.drop(columns=constant_cols)

    removed_vif = []
    while True:
        vif_vals = [variance_inflation_factor(X.values.astype(float), i)
                    for i in range(X.shape[1])]
        vif_df = pd.DataFrame({'feature': X.columns, 'VIF': vif_vals})
        max_vif = vif_df['VIF'].max()
        if max_vif < 10:
            break
        worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        removed_vif.append(worst)
        print(f'  VIF Excluded: {worst} (VIF={max_vif:.2f})')
        X = X.drop(columns=[worst])

    keep = X.columns.tolist()
    print(f'  VIF Filtering Complete: Kept {len(keep)} features, Removed {len(removed_vif)}')
    if removed_vif:
        print(f'  VIF Excluded List: {removed_vif}')
    print(f'  Final Feature Space: {keep}')

    return X[keep], y, df, keep

# ===================================================================
# Step 2: Hyperparameter Optimization via Optuna
# ===================================================================
def _tscv_r2(model_cls, params, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        m = model_cls(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(scores))

def tune_rf(X, y, n_trials):
    print('  Optimizing Random Forest Params...')
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_float('max_features', 0.3, 1.0),
            random_state=42, n_jobs=-1)
        return _tscv_r2(RandomForestRegressor, p, X, y)
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    p = s.best_params; p.update({'random_state': 42, 'n_jobs': -1})
    print(f'    Best CV-R2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

def tune_xgb(X, y, n_trials):
    print('  Optimizing XGBoost Params...')
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            random_state=42, verbosity=0)
        return _tscv_r2(xgb.XGBRegressor, p, X, y)
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    p = s.best_params; p.update({'random_state': 42, 'verbosity': 0})
    print(f'    Best CV-R2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

def tune_lgb(X, y, n_trials):
    print('  Optimizing LightGBM Params...')
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            num_leaves=trial.suggest_int('num_leaves', 20, 150),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            random_state=42, verbose=-1)
        return _tscv_r2(lgb.LGBMRegressor, p, X, y)
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    p = s.best_params; p.update({'random_state': 42, 'verbose': -1})
    print(f'    Best CV-R2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

# ===================================================================
# Step 3: Model Evaluation (Time-Series Cross Validation)
# ===================================================================
def evaluate_model(model, X, y, model_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = MinMaxScaler()
    rows, all_pred, all_true = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        Xtr = pd.DataFrame(scaler.fit_transform(X.iloc[tr]), columns=X.columns)
        Xte = pd.DataFrame(scaler.transform(X.iloc[te]), columns=X.columns)
        ytr, yte = y.iloc[tr], y.iloc[te]
        m = model.__class__(**model.get_params())
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        all_pred.extend(pred); all_true.extend(yte.values)
        rows.append({'Fold': fold,
                     'R2': r2_score(yte, pred),
                     'RMSE': np.sqrt(mean_squared_error(yte, pred)),
                     'MAE': mean_absolute_error(yte, pred)})

    fold_df = pd.DataFrame(rows)
    overall = {'Model': model_name,
               'R2_mean': fold_df['R2'].mean(), 'R2_std': fold_df['R2'].std(),
               'RMSE_mean': fold_df['RMSE'].mean(), 'MAE_mean': fold_df['MAE'].mean()}
    print(f"  {model_name:15s}  R2={overall['R2_mean']:.4f}+-{overall['R2_std']:.4f}"
          f"  RMSE={overall['RMSE_mean']:.2f}  MAE={overall['MAE_mean']:.2f}")
    return fold_df, overall, np.array(all_true), np.array(all_pred)

# ===================================================================
# Step 4: Visualization (Model Validation Performance Comparison)
# ===================================================================
def plot_model_comparison(metrics_list, out_dir):
    df_m = pd.DataFrame(metrics_list)
    models = df_m['Model'].tolist()
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, label, fmt) in zip(axes, [
            ('R2_mean', 'Mean R2', '.4f'),
            ('RMSE_mean', 'Mean RMSE', '.2f'),
            ('MAE_mean', 'Mean MAE', '.2f')]):
        vals = df_m[col].values
        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)

    plt.suptitle(f'{GAS} - Model Performance Comparison (5-Fold Time-Series CV)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig1_model_comparison.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f'  Saved -> {p}')

def plot_fold_r2(fold_results, out_dir):
    colors = {'Random Forest': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, fd in fold_results.items():
        ax.plot(fd['Fold'], fd['R2'], marker='o', linewidth=2,
                label=name, color=colors[name])
    ax.set_xlabel('Fold', fontsize=11); ax.set_ylabel('R2', fontsize=11)
    ax.set_title(f'{GAS} - R2 by Fold (Time-Series CV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig2_fold_r2.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f'  Saved -> {p}')

def save_metrics_excel(metrics_list, fold_results, out_dir):
    p = os.path.join(out_dir, 'model_metrics_summary.xlsx')
    with pd.ExcelWriter(p, engine='openpyxl') as w:
        pd.DataFrame(metrics_list).to_excel(w, sheet_name='Overall', index=False)
        for name, fd in fold_results.items():
            fd.to_excel(w, sheet_name=name.replace(' ', '_')[:31], index=False)
    print(f'  Saved Summary Excel -> {p}')

# ===================================================================
# Step 5: Global SHAP Explanations
# ===================================================================
def _dim_colors_for(feats):
    return [DIM_COLORS.get(FEATURE_DIM.get(f, 'Economy'), '#999') for f in feats]

def plot_global_shap(shap_dict, X_sc_dict, out_dir, top_n=15):
    all_global_rows = []

    for name, sv in shap_dict.items():
        Xsc = X_sc_dict[name]
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xsc.columns)
        top_n_actual = min(top_n, len(mean_abs))
        top_f = mean_abs.nlargest(top_n_actual).index.tolist()[::-1]
        idx = [Xsc.columns.get_loc(f) for f in top_f]
        safe = name.replace(' ', '_')

        # Summary Beeswarm Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv[:, idx], Xsc[top_f], show=False, plot_size=None)
        ax = plt.gca()
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(lbl, 'Economy'), '#333'))
            tick.set_fontsize(10)
        ax.set_title(f'{name} ({GAS}) - Global SHAP Beeswarm (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3_beeswarm_{safe}.png'),
                    dpi=300, bbox_inches='tight'); plt.close()

        # Summary Violin Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv[:, idx], Xsc[top_f], plot_type='violin',
                          show=False, plot_size=None)
        ax = plt.gca()
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(lbl, 'Economy'), '#333'))
            tick.set_fontsize(10)
        ax.set_title(f'{name} ({GAS}) - Global SHAP Violin (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3b_violin_{safe}.png'),
                    dpi=300, bbox_inches='tight'); plt.close()

        # Feature Importance Bar Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_f, mean_abs[top_f], color=_dim_colors_for(top_f), edgecolor='white')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
        ax.set_title(f'{name} ({GAS}) - Feature Importance (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig4_bar_{safe}.png'),
                    dpi=300, bbox_inches='tight'); plt.close()

        # Global Metric Tables
        total_shap = mean_abs.sum()
        df_table = pd.DataFrame({
            'model': name,
            'feature': mean_abs.index,
            'mean_abs_shap': mean_abs.values,
            'shap_ratio': mean_abs.values / total_shap,
            'dimension': [FEATURE_DIM.get(f, 'Unknown') for f in mean_abs.index],
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        df_table['rank'] = df_table.index + 1
        all_global_rows.append(df_table)
        df_table.to_excel(
            os.path.join(out_dir, f'global_shap_table_{safe}.xlsx'), index=False)

    path_all = os.path.join(out_dir, 'global_shap_table_all_models.xlsx')
    with pd.ExcelWriter(path_all, engine='openpyxl') as w:
        for df_t in all_global_rows:
            sheet = df_t['model'].iloc[0].replace(' ', '_')[:31]
            df_t.to_excel(w, sheet_name=sheet, index=False)
    print(f'  Saved Global Explanations (Beeswarm, Violin, Bar charts)')
    print(f'  Summary Table -> {path_all}')

# ===================================================================
# Step 6: Multi-Model SHAP Divergence Check
# ===================================================================
def plot_shap_comparison(shap_dict, X_sc_dict, out_dir, top_n=15):
    base_name = 'XGBoost' if 'XGBoost' in shap_dict else list(shap_dict.keys())[0]
    n_feats = len(X_sc_dict[base_name].columns)
    top_n_actual = min(top_n, n_feats)
    base_feats = (pd.Series(np.abs(shap_dict[base_name]).mean(axis=0),
                            index=X_sc_dict[base_name].columns)
                  .nlargest(top_n_actual).index.tolist())

    records = {}
    for name, sv in shap_dict.items():
        ma = pd.Series(np.abs(sv).mean(axis=0), index=X_sc_dict[name].columns)
        records[name] = [ma.get(f, 0) for f in base_feats]
    df_cmp = pd.DataFrame(records, index=base_feats)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(base_feats))
    w = 0.25
    clr = {'Random Forest': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}
    for i, (name, vals) in enumerate(df_cmp.items()):
        ax.barh(x + (i-1)*w, vals, height=w, label=name,
                color=clr.get(name, '#999'), edgecolor='white', alpha=0.9)
    ax.set_yticks(x); ax.set_yticklabels(base_feats, fontsize=9)
    for tick, feat in zip(ax.get_yticklabels(), base_feats):
        tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(feat, 'Economy'), '#333'))
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title(f'{GAS} - Top {top_n_actual} Features: SHAP Comparison Across Models',
                 fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    dim_patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
    mdl_patches = [mpatches.Patch(color=clr[n], label=n) for n in clr]
    ax.legend(handles=dim_patches + mdl_patches, fontsize=8,
              loc='lower right', framealpha=0.8, ncol=2)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig5_shap_3model_comparison.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f'  Saved Cross-Model Diagnostics Plot -> {p}')

    df_cmp.to_excel(os.path.join(out_dir, 'shap_comparison_table.xlsx'))
    return df_cmp

# ===================================================================
# Step 7: Spatial/Regional SHAP Slicing
# ===================================================================
def plot_regional_shap(best_model, X_scaled, df_raw, out_dir, top_n=10):
    rdir = os.path.join(out_dir, 'regional_shap')
    os.makedirs(rdir, exist_ok=True)

    Xsc = X_scaled.copy()
    Xsc['_region'] = df_raw['region'].values
    top_n_actual = min(top_n, X_scaled.shape[1])

    long_rows = []
    for region in sorted(Xsc['_region'].unique()):
        mask = Xsc['_region'] == region
        X_reg = Xsc.loc[mask, X_scaled.columns]

        if len(X_reg) < 2:
            continue

        bg_size = min(len(X_reg), 100)
        bg = shap.sample(X_reg, bg_size, random_state=42)
        explainer = shap.TreeExplainer(best_model, data=bg,
                               Feature_perturbation='interventional')
        try:
            sv = explainer.shap_values(X_reg, check_additivity=False)
            if isinstance(sv, list): sv = sv[0]
        except Exception as e:
            print(f'  Warning: Regional partition {region} analysis failed: {e}')
            continue

        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_reg.columns)
        total = mean_abs.sum()
        top_feats = mean_abs.nlargest(top_n_actual).index.tolist()[::-1]
        idx = [X_reg.columns.get_loc(f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv[:, idx], X_reg[top_feats], show=False, plot_size=None)
        ax = plt.gca()
        ax.set_title(f'{GAS} Region: {region} - SHAP (Top {top_n_actual})',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(rdir, f'regional_{region}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

        for feat, val in mean_abs.items():
            long_rows.append({
                'region': region,
                'feature': feat,
                'mean_abs_shap': val,
                'shap_ratio': val / total if total > 0 else 0,
                'dimension': FEATURE_DIM.get(feat, 'Unknown'),
            })

    df_long = pd.DataFrame(long_rows).sort_values(
        ['region', 'mean_abs_shap'], ascending=[True, False])

    df_long.to_excel(os.path.join(out_dir, 'regional_shap_long.xlsx'), index=False)
    df_wide = df_long.pivot(index='region', columns='feature', values='mean_abs_shap')
    df_wide.to_excel(os.path.join(out_dir, 'regional_shap_wide.xlsx'))

    path_by_region = os.path.join(out_dir, 'regional_shap_by_region.xlsx')
    with pd.ExcelWriter(path_by_region, engine='openpyxl') as w:
        for region in df_long['region'].unique():
            sub = df_long[df_long['region'] == region].reset_index(drop=True)
            sub['rank'] = sub.index + 1
            sub.to_excel(w, sheet_name=str(region).replace('/', '-')[:31], index=False)

    print(f'  Saved Regional Explanations (Partitions evaluated: {df_long["region"].nunique()})')
    return df_long

# ===================================================================
# Main Orchestration Loop
# ===================================================================
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Data Pipeline
    X, y, df_raw, feat_cols = prepare_data(FILE)

    # Step 2: Optimization Engine
    print('\n' + '='*60)
    print(f'[Step 2] Parameter Optimization Loop ({N_TRIALS} Trials Per Estimator)')
    rf_params, rf_r2 = tune_rf(X, y, N_TRIALS)
    xgb_params, xgb_r2 = tune_xgb(X, y, N_TRIALS)
    lgb_params, lgb_r2 = tune_lgb(X, y, N_TRIALS)

    # Step 3: Model Cross-Validation
    print('\n' + '='*60)
    print('[Step 3] Time-Series Empirical Validation (5-Fold Split)')
    models = {
        'Random Forest': RandomForestRegressor(**rf_params),
        'XGBoost': xgb.XGBRegressor(**xgb_params),
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
    }
    metrics_list, fold_results = [], {}
    for name, model in models.items():
        fd, ov, true_vals, pred_vals = evaluate_model(model, X, y, name)
        metrics_list.append(ov); fold_results[name] = fd
        pred_df = pd.DataFrame({'y_true': true_vals, 'y_pred': pred_vals})
        pred_df.to_excel(
            os.path.join(OUT_DIR, f'cv_predictions_{name.replace(" ","_")}.xlsx'),
            index=False)

    # Step 4: Metric Plots
    print('\n' + '='*60)
    print('[Step 4] Storing Model Diagnostics Tables and Charts')
    plot_model_comparison(metrics_list, OUT_DIR)
    plot_fold_r2(fold_results, OUT_DIR)
    save_metrics_excel(metrics_list, fold_results, OUT_DIR)

    # Step 5: Full-Fit Training and SHAP Computation
    print('\n' + '='*60)
    print('[Step 5] Estimator Fitting Over Full Training Matrix')
    scaler = MinMaxScaler()
    Xsc_all = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    shap_dict, Xsc_dict = {}, {}
    best_model, best_r2_val, best_name = None, -np.inf, ''

    for name, model in models.items():
        model.fit(Xsc_all, y)
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(Xsc_all)
        if isinstance(sv, list): sv = sv[0]
        shap_dict[name] = sv
        Xsc_dict[name] = Xsc_all.copy()
        r2 = next(m['R2_mean'] for m in metrics_list if m['Model'] == name)
        if r2 > best_r2_val:
            best_r2_val, best_model, best_name = r2, model, name

    print(f'  Selected Optimal Estimator: {best_name} (Cross-Validated R2={best_r2_val:.4f})')

    # Step 6: Global Importance Profiling
    print('\n' + '='*60)
    print('[Step 6] Compiling Structural Importance Profiles')
    plot_global_shap(shap_dict, Xsc_dict, OUT_DIR, top_n=15)
    plot_shap_comparison(shap_dict, Xsc_dict, OUT_DIR, top_n=15)

    # Step 7: Spatial Profiling
    print('\n' + '='*60)
    print(f'[Step 7] Running Sub-Regional Slicing Engine ({best_name})')
    plot_regional_shap(best_model, Xsc_all, df_raw, OUT_DIR, top_n=10)

    print('\n' + '='*60)
    print(f'Pipeline completed without exceptions. Workspace target directory: {OUT_DIR}')
    print('='*60)
