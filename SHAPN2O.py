import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

FILE = 'X'
GAS = 'N2O'
OUT_DIR = 'X'
N_TRIALS = 50

DIM_COLORS = {
    'Economy': '#2196F3',
    'Society': '#4CAF50',
    'Industry': '#FF9800',
    'Technology': '#9C27B0',
    'Policy': '#F44336',
    'Environment': '#00BCD4',
}

FEATURE_DIM = {
    'GDP': 'Economy', 'PGDP': 'Economy', 'CCI': 'Economy', 'OAG': 'Economy', 'APEV': 'Economy',
    'POP': 'Society', 'URB': 'Society', 'DS': 'Society',
    'POV': 'Industry', 'LOV': 'Industry', 'IVA': 'Industry',
    'POVP': 'Industry', 'LOVP': 'Industry', 'IOVP': 'Industry',
    'ALA': 'Industry', 'ANFUI': 'Industry',

    'CRPS': 'Technology', 'CRPP': 'Technology',
    'COMS': 'Technology', 'COMP': 'Technology',
    'LIVS': 'Technology', 'LIVP': 'Technology',
    'WSTS': 'Technology', 'WSTP': 'Technology',
    'INDS': 'Technology', 'INDP': 'Technology',

    'POL': 'Policy',
    'TEMP': 'Environment', 'PRCP': 'Environment',
}

def prepare_data(file_path):
    print('=' * 60)
    print('\u3010Step 1\u3011\u52a0\u8f7d\u6570\u636e & \u7279\u5f81\u6784\u9020')

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values('Year').reset_index(drop=True)
    print(f'  \u539f\u59cb\u5217\u540d: {list(df.columns)}')

    col_map = {
        '\u964d\u6c34': 'PRCP',
        '\u6c14\u6e29': 'TEMP',
        '\u4eba\u53e3': 'POP',
        'GDP': 'GDP',
        '\u4eba\u5747GDP': 'PGDP',
        '\u57ce\u9547\u5316\u7387': 'URB',
        '\u519c\u4e1a\u6c2e\u80a5\u7528\u91cf': 'ANFU',
        '\u519c\u4e1a\u7528\u5730\u9762\u79ef': 'ALA',
        '\u79cd\u690d\u4e1a\u4ea7\u503c': 'POV',
        '\u755c\u7267\u4e1a\u4ea7\u503c': 'LOV',
        '\u5de5\u4e1a\u4ea7\u503c': 'IVA',
        '\u996e\u98df\u7ed3\u6784': 'DS',
        '\u653f\u7b56\u6570\u91cf': 'POL',
        '\u519c\u4ea7\u54c1\u51fa\u53e3\u989d': 'APEV',

        'Croplands (agricultural soils)_\u6280\u672f\u89c4\u6a21': 'CRPS',
        'Croplands (agricultural soils)_\u6280\u672f\u6e17\u900f\u7387': 'CRPP',
        'Combustion of fossil fuels and biomass_\u6280\u672f\u89c4\u6a21': 'COMS',
        'Combustion of fossil fuels and biomass_\u6280\u672f\u6e17\u900f\u7387': 'COMP',
        'Livestock\uff08Enteric fermentation and Manure management\uff09_\u6280\u672f\u89c4\u6a21': 'LIVS',
        'Livestock\uff08Enteric fermentation and Manure management\uff09_\u6280\u672f\u6e17\u900f\u7387': 'LIVP',
        'Waste_\u6280\u672f\u89c4\u6a21': 'WSTS',
        'Waste_\u6280\u672f\u6e17\u900f\u7387': 'WSTP',
        'Industrial Processes_\u6280\u672f\u89c4\u6a21': 'INDS',
        'Industrial Processes_\u6280\u672f\u6e17\u900f\u7387': 'INDP',

    }
    df = df.rename(columns=col_map)

    eps = 1e-9

    if '\u7164\u70ad\u6d88\u8d39\u91cf' in df.columns:
        df['CCI'] = df['\u7164\u70ad\u6d88\u8d39\u91cf'] / (df['GDP'] + eps)
        print('  \u2713 \u521d\u59cb CCI \u6784\u9020\u6210\u529f')
    else:
        print('  \u26a0 \u6587\u4ef6\u4e2d\u65e0"\u7164\u70ad\u6d88\u8d39\u91cf"\u5217\uff0cCCI \u8df3\u8fc7')

    if '\u77f3\u6cb9\u6d88\u8d39\u91cf' in df.columns and '\u5929\u7136\u6c14\u6d88\u8d39\u91cf' in df.columns:
        df['OAG'] = df['\u77f3\u6cb9\u6d88\u8d39\u91cf'] + df['\u5929\u7136\u6c14\u6d88\u8d39\u91cf']
        print('  \u2713 \u539f\u59cb\u6cb9\u6c14\u6d88\u8d39\u91cf OAG (\u77f3\u6cb9+\u5929\u7136\u6c14) \u63d0\u53d6\u6210\u529f')
    elif '\u77f3\u6cb9\u6d88\u8d39\u91cf' in df.columns:
        df['OAG'] = df['\u77f3\u6cb9\u6d88\u8d39\u91cf']
        print('  \u2713 \u539f\u59cb\u6cb9\u6c14\u6d88\u8d39\u91cf OAG (\u4ec5\u77f3\u6cb9) \u63d0\u53d6\u6210\u529f')
    elif '\u6cb9\u6c14\u6d88\u8d39\u91cf' in df.columns:
        df['OAG'] = df['\u6cb9\u6c14\u6d88\u8d39\u91cf']
        print('  \u2713 \u539f\u59cb\u6cb9\u6c14\u6d88\u8d39\u91cf OAG \u63d0\u53d6\u6210\u529f')
    else:
        print('  \u26a0 \u6587\u4ef6\u4e2d\u65e0\u77f3\u6cb9/\u5929\u7136\u6c14/\u6cb9\u6c14\u6d88\u8d39\u91cf\u5217\uff0cOAG \u63d0\u53d6\u8df3\u8fc7')

    control_vars = ['Year', 'PGDP', 'URB']

    if all(col in df.columns for col in control_vars):
        X_control = df[control_vars].copy()
        X_control = X_control.replace([np.inf, -np.inf], np.nan)
        for col in control_vars:
            X_control[col] = X_control[col].fillna(X_control[col].median())

        target_residuals = ['DS', 'CCI']
        for target in target_residuals:
            if target in df.columns:
                y_target = df[target].replace([np.inf, -np.inf], np.nan)
                y_target = y_target.fillna(y_target.median())

                lr = LinearRegression()
                lr.fit(X_control, y_target)

                df[target] = y_target - lr.predict(X_control)
                print(f'  \u2713 {target} \u5df2\u6210\u529f\u66ff\u6362\u4e3a\u63a7\u5236 [Year, PGDP, URB] \u540e\u7684\u6b8b\u5dee\u9879 ({target}_resid)')
            else:
                print(f'  \u26a0 \u6570\u636e\u4e2d\u7f3a\u5931 {target}\uff0c\u65e0\u6cd5\u8fdb\u884c\u6b8b\u5dee\u5316\u5904\u7406')
    else:
        missing_controls = [c for c in control_vars if c not in df.columns]
        print(f'  \u26a0 \u7f3a\u5c11\u63a7\u5236\u53d8\u91cf {missing_controls}\uff0c\u8df3\u8fc7\u6b8b\u5dee\u5904\u7406\u6b65\u9aa4\uff01')

    if 'ANFU' in df.columns and 'ALA' in df.columns:
        df['ANFUI'] = df['ANFU'] / (df['ALA'] + eps)
        print('  \u2713 ANFUI (\u6c2e\u80a5\u5f3a\u5ea6) \u6784\u9020\u6210\u529f')

    if 'POV' in df.columns:
        df['POVP'] = df['POV'] / (df['GDP'] + eps)
    if 'LOV' in df.columns:
        df['LOVP'] = df['LOV'] / (df['GDP'] + eps)
    if 'IVA' in df.columns:
        df['IOVP'] = df['IVA'] / (df['GDP'] + eps)

    n2o_features = [
        'GDP', 'PGDP', 'CCI', 'OAG', 'APEV',
        'POP', 'URB', 'DS',
        'POV', 'LOV', 'IVA',
        'POVP', 'LOVP', 'IOVP',
        'ALA', 'ANFUI',
        'CRPS', 'CRPP',
        'COMS', 'COMP',
        'LIVS', 'LIVP',
        'WSTS', 'WSTP',
        'INDS', 'INDP',
        'POL',
        'TEMP', 'PRCP',
    ]

    feature_cols = [c for c in n2o_features if c in df.columns]
    missing = set(n2o_features) - set(feature_cols)
    if missing:
        print(f'  \u26a0 \u4ee5\u4e0b\u7279\u5f81\u4e0d\u53ef\u7528\uff08\u6587\u4ef6\u65e0\u539f\u59cb\u5217\u6216\u65e0\u6cd5\u6784\u9020\uff09: {sorted(missing)}')
    print(f'  \u5b9e\u9645\u4f7f\u7528\u7279\u5f81 ({len(feature_cols)}): {feature_cols}')

    X = df[feature_cols].copy()
    y = df['\u6392\u653e\u91cf'].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        print(f'  \u79fb\u9664\u5e38\u91cf\u5217: {constant_cols}')
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
        print(f'  VIF\u79fb\u9664: {worst} (VIF={max_vif:.2f})')
        X = X.drop(columns=[worst])

    keep = X.columns.tolist()
    print(f'  VIF\u7b5b\u9009\u5b8c\u6210: \u4fdd\u7559 {len(keep)} \u4e2a\u7279\u5f81\uff0c\u5171\u79fb\u9664 {len(removed_vif)} \u4e2a')
    if removed_vif:
        print(f'  VIF\u79fb\u9664\u5217\u8868: {removed_vif}')
    print(f'  \u6700\u7ec8\u7279\u5f81: {keep}')

    return X[keep], y, df, keep

def _tscv_r2(model_cls, params, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        m = model_cls(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(scores))

def tune_rf(X, y, n_trials):
    print('  \u8c03\u53c2 Random Forest ...')

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
    p = s.best_params
    p.update({'random_state': 42, 'n_jobs': -1})
    print(f'    \u6700\u4f18CV-R\u00b2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

def tune_xgb(X, y, n_trials):
    print('  \u8c03\u53c2 XGBoost ...')

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
    p = s.best_params
    p.update({'random_state': 42, 'verbosity': 0})
    print(f'    \u6700\u4f18CV-R\u00b2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

def tune_lgb(X, y, n_trials):
    print('  \u8c03\u53c2 LightGBM ...')

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
    p = s.best_params
    p.update({'random_state': 42, 'verbose': -1})
    print(f'    \u6700\u4f18CV-R\u00b2: {s.best_value:.4f}  params: {p}')
    return p, s.best_value

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
        all_pred.extend(pred)
        all_true.extend(yte.values)
        rows.append({'Fold': fold,
                     'R\u00b2': r2_score(yte, pred),
                     'RMSE': np.sqrt(mean_squared_error(yte, pred)),
                     'MAE': mean_absolute_error(yte, pred)})

    fold_df = pd.DataFrame(rows)
    overall = {'Model': model_name,
               'R\u00b2_mean': fold_df['R\u00b2'].mean(), 'R\u00b2_std': fold_df['R\u00b2'].std(),
               'RMSE_mean': fold_df['RMSE'].mean(), 'MAE_mean': fold_df['MAE'].mean()}
    print(f"  {model_name:15s}  R\u00b2={overall['R\u00b2_mean']:.4f}\u00b1{overall['R\u00b2_std']:.4f}"
          f"  RMSE={overall['RMSE_mean']:.2f}  MAE={overall['MAE_mean']:.2f}")
    return fold_df, overall, np.array(all_true), np.array(all_pred)

def plot_model_comparison(metrics_list, out_dir):
    df_m = pd.DataFrame(metrics_list)
    models = df_m['Model'].tolist()
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, label, fmt) in zip(axes, [
        ('R\u00b2_mean', 'Mean R\u00b2', '.4f'),
        ('RMSE_mean', 'Mean RMSE', '.2f'),
        ('MAE_mean', 'Mean MAE', '.2f')]):
        vals = df_m[col].values
        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)

    plt.suptitle(f'{GAS} \u2014 Model Performance Comparison (5-Fold Time-Series CV)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig1_model_comparison.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  \u5df2\u4fdd\u5b58 \u2192 {p}')

def plot_fold_r2(fold_results, out_dir):
    colors = {'Random Forest': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, fd in fold_results.items():
        ax.plot(fd['Fold'], fd['R\u00b2'], marker='o', linewidth=2,
                label=name, color=colors[name])
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('R\u00b2', fontsize=11)
    ax.set_title(f'{GAS} \u2014 R\u00b2 by Fold (Time-Series CV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig2_fold_r2.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  \u5df2\u4fdd\u5b58 \u2192 {p}')

def save_metrics_excel(metrics_list, fold_results, out_dir):
    p = os.path.join(out_dir, 'model_metrics_summary.xlsx')
    with pd.ExcelWriter(p, engine='openpyxl') as w:
        pd.DataFrame(metrics_list).to_excel(w, sheet_name='Overall', index=False)
        for name, fd in fold_results.items():
            fd.to_excel(w, sheet_name=name.replace(' ', '_')[:31], index=False)
    print(f'  \u5df2\u4fdd\u5b58\u6307\u6807\u8868 \u2192 {p}')

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

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv[:, idx], Xsc[top_f], show=False, plot_size=None)
        ax = plt.gca()
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(lbl, 'Economy'), '#333'))
            tick.set_fontsize(10)
        ax.set_title(f'{name} ({GAS}) \u2014 Global SHAP Beeswarm (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3_beeswarm_{safe}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv[:, idx], Xsc[top_f], plot_type='violin',
                          show=False, plot_size=None)
        ax = plt.gca()
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(lbl, 'Economy'), '#333'))
            tick.set_fontsize(10)
        ax.set_title(f'{name} ({GAS}) \u2014 Global SHAP Violin (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3b_violin_{safe}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_f, mean_abs[top_f], color=_dim_colors_for(top_f), edgecolor='white')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
        ax.set_title(f'{name} ({GAS}) \u2014 Feature Importance (Top {top_n_actual})',
                     fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8,
                  loc='lower right', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig4_bar_{safe}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

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
    print(f'  \u5df2\u4fdd\u5b58\u5168\u5c40SHAP\u8702\u7a9d\u56fe\u3001\u5c0f\u63d0\u7434\u56fe\u3001\u6761\u5f62\u56fe & \u6570\u636e\u8868\uff08\u4e09\u6a21\u578b\u5404\u4e00\u5957\uff09')
    print(f'  \u6c47\u603b\u6570\u636e\u8868 \u2192 {path_all}')

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
        ax.barh(x + (i - 1) * w, vals, height=w, label=name,
                color=clr.get(name, '#999'), edgecolor='white', alpha=0.9)
    ax.set_yticks(x)
    ax.set_yticklabels(base_feats, fontsize=9)
    for tick, feat in zip(ax.get_yticklabels(), base_feats):
        tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(feat, 'Economy'), '#333'))
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title(f'{GAS} \u2014 Top {top_n_actual} Features: SHAP Comparison Across Models',
                 fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    dim_patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
    mdl_patches = [mpatches.Patch(color=clr[n], label=n) for n in clr]
    ax.legend(handles=dim_patches + mdl_patches, fontsize=8,
              loc='lower right', framealpha=0.8, ncol=2)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig5_shap_3model_comparison.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  \u5df2\u4fdd\u5b58 \u2192 {p}')

    df_cmp.to_excel(os.path.join(out_dir, 'shap_comparison_table.xlsx'))
    return df_cmp

def plot_regional_shap(best_model, X_scaled, df_raw, out_dir, top_n=10):
    rdir = os.path.join(out_dir, 'regional_shap')
    os.makedirs(rdir, exist_ok=True)

    Xsc = X_scaled.copy()
    Xsc['_region'] = df_raw['region'].values
    top_n_actual = min(top_n, X_scaled.shape[1])

    long_rows = []
    matrix_rows = []

    for region in sorted(Xsc['_region'].unique()):
        mask = Xsc['_region'] == region
        X_reg = Xsc.loc[mask, X_scaled.columns]

        if len(X_reg) < 2:
            continue

        bg_size = min(len(X_reg), 100)
        bg = shap.sample(X_reg, bg_size, random_state=42)
        explainer = shap.TreeExplainer(best_model, data=bg,
                                       feature_perturbation='interventional')
        try:
            sv = explainer.shap_values(X_reg, check_additivity=False)
            if isinstance(sv, list): sv = sv[0]
        except Exception as e:
            print(f'  \u26a0 \u533a\u57df {region} \u5206\u6790\u5931\u8d25: {e}')
            continue

        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_reg.columns)
        mean_signed = pd.Series(sv.mean(axis=0), index=X_reg.columns)
        total = mean_abs.sum()

        region_matrix_dict = {'region': region}
        for feat in X_scaled.columns:
            region_matrix_dict[feat] = mean_signed.get(feat, 0.0)
        matrix_rows.append(region_matrix_dict)

        top_feats = mean_abs.nlargest(top_n_actual).index.tolist()[::-1]
        idx = [X_reg.columns.get_loc(f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv[:, idx], X_reg[top_feats], show=False, plot_size=None)
        ax = plt.gca()
        ax.set_title(f'{GAS} Region: {region} \u2014 SHAP (Top {top_n_actual})',
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

    df_matrix = pd.DataFrame(matrix_rows).set_index('region')
    path_matrix = os.path.join(out_dir, 'regional_shap_mean_matrix.xlsx')
    df_matrix.to_excel(path_matrix)

    print(f'  \u5df2\u4fdd\u5b58\u533a\u57dfSHAP\u56fe\uff08{df_long["region"].nunique()}\u4e2a\u533a\u57df\uff09')
    print(f'  \u957f\u8868 \u2192 regional_shap_long.xlsx')
    print(f'  \u5bbd\u8868 \u2192 regional_shap_wide.xlsx')
    print(f'  \u5206\u533a\u57df\u8868 \u2192 regional_shap_by_region.xlsx')
    print(f'  \u2713 \u6210\u529f\u8f93\u51fa\u539f\u59cb\u5e26\u7b26\u53f7\u7684\u65b9\u5411\u6027\u5747\u503c\u77e9\u9635 \u2192 regional_shap_mean_matrix.xlsx')
    return df_long

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, df_raw, feat_cols = prepare_data(FILE)

    print('\n' + '=' * 60)
    print(f'\u3010Step 2\u3011Optuna \u8d85\u53c2\u6570\u4f18\u5316\uff08\u6bcf\u6a21\u578b {N_TRIALS} \u6b21\u8bd5\u9a8c\uff09')
    rf_params, rf_r2 = tune_rf(X, y, N_TRIALS)
    xgb_params, xgb_r2 = tune_xgb(X, y, N_TRIALS)
    lgb_params, lgb_r2 = tune_lgb(X, y, N_TRIALS)

    print('\n' + '=' * 60)
    print('\u3010Step 3\u3011\u65f6\u95f4\u5e8f\u5217 5 \u6298\u4ea4\u53c9\u9a8c\u8bc1')
    models = {
        'Random Forest': RandomForestRegressor(**rf_params),
        'XGBoost': xgb.XGBRegressor(**xgb_params),
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
    }
    metrics_list, fold_results = [], {}
    for name, model in models.items():
        fd, ov, true_vals, pred_vals = evaluate_model(model, X, y, name)
        metrics_list.append(ov)
        fold_results[name] = fd
        pred_df = pd.DataFrame({'y_true': true_vals, 'y_pred': pred_vals})
        pred_df.to_excel(
            os.path.join(OUT_DIR, f'cv_predictions_{name.replace(" ", "_")}.xlsx'),
            index=False)

    print('\n' + '=' * 60)
    print('\u3010Step 4\u3011\u4fdd\u5b58\u6a21\u578b\u6027\u80fd\u56fe\u8868')
    plot_model_comparison(metrics_list, OUT_DIR)
    plot_fold_r2(fold_results, OUT_DIR)
    save_metrics_excel(metrics_list, fold_results, OUT_DIR)

    print('\n' + '=' * 60)
    print('\u3010Step 5\u3011\u5168\u91cf\u6570\u636e\u8bad\u7ec3 + SHAP \u5206\u6790')
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
        r2 = next(m['R\u00b2_mean'] for m in metrics_list if m['Model'] == name)
        if r2 > best_r2_val:
            best_r2_val, best_model, best_name = r2, model, name

    print(f'  \u6700\u4f18\u6a21\u578b: {best_name}  (CV R\u00b2={best_r2_val:.4f})')

    print('\n' + '=' * 60)
    print('\u3010Step 6\u3011\u5168\u5c40 SHAP \u56fe\uff08\u4e09\u6a21\u578b\uff09')
    plot_global_shap(shap_dict, Xsc_dict, OUT_DIR, top_n=15)
    plot_shap_comparison(shap_dict, Xsc_dict, OUT_DIR, top_n=15)

    print('\n' + '=' * 60)
    print(f'\u3010Step 7\u3011\u533a\u57df SHAP \u5206\u6790\uff08{best_name}\uff09')
    plot_regional_shap(best_model, Xsc_all, df_raw, OUT_DIR, top_n=10)

    print('\n' + '=' * 60)
    print(f'\U0001f389  {GAS} \u5168\u6d41\u7a0b\u5b8c\u6210\uff01\u8f93\u51fa\u76ee\u5f55: {OUT_DIR}')
    print('=' * 60)
