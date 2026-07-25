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
FILE = os.path.join('X', 'CH4_filled.xlsx')
GAS = 'CH4'
OUT_DIR = os.path.join('X', f'output_SHAP_{GAS}')
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

    'GDP': 'Economy', 'PGDP': 'Economy', 'CCI': 'Economy',

    'OAG': 'Economy', 'APEV': 'Economy',

    'POP': 'Society', 'URB': 'Society', 'DS': 'Society',

    'POV': 'Industry', 'LOV': 'Industry', 'ALA': 'Industry',

    'RPA': 'Industry', 'IVA': 'Industry', 'IOVP': 'Industry',

    'POVP': 'Industry', 'LOVP': 'Industry',

    'LIVS': 'Technology', 'LIVP': 'Technology',

    'CRPS': 'Technology', 'CRPP': 'Technology',

    'RICS': 'Technology', 'RICP': 'Technology',

    'COMS': 'Technology', 'COMP': 'Technology',

    'GASS': 'Technology', 'GASP': 'Technology',

    'CLMS': 'Technology', 'CLMP': 'Technology',

    'WSTS': 'Technology', 'WSTP': 'Technology',

    'INDS': 'Technology', 'INDP': 'Technology',

    'POL': 'Policy',

    'TEMP': 'Environment', 'PRCP': 'Environment',

}

def prepare_data(file_path):

    print('=' * 60)

    print('Step 1: Load data and construct features')

    df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()

    df = df.sort_values('Year').reset_index(drop=True)

    print(f'  Number of original columns: {len(df.columns)}')

    col_map = {

        '\u964d\u6c34': 'PRCP',

        '\u6c14\u6e29': 'TEMP',

        '\u4eba\u53e3': 'POP',

        'GDP': 'GDP',

        '\u4eba\u5747GDP': 'PGDP',

        '\u57ce\u9547\u5316\u7387': 'URB',

        '\u519c\u4ea7\u54c1\u51fa\u53e3\u989d': 'APEV',

        '\u519c\u4e1a\u7528\u5730\u9762\u79ef': 'ALA',

        '\u6c34\u7a3b\u79cd\u690d\u9762\u79ef': 'RPA',

        '\u79cd\u690d\u4e1a\u4ea7\u503c': 'POV',

        '\u755c\u7267\u4e1a\u4ea7\u503c': 'LOV',

        '\u996e\u98df\u7ed3\u6784': 'DS',

        '\u653f\u7b56\u6570\u91cf': 'POL',

        '\u5de5\u4e1a\u4ea7\u503c': 'IVA',

        'Livestock\uff08Enteric fermentation and Manure management\uff09_\u6280\u672f\u89c4\u6a21': 'LIVS',

        'Livestock\uff08Enteric fermentation and Manure management\uff09_\u6280\u672f\u6e17\u900f\u7387': 'LIVP',

        'Croplands (agricultural soils)_\u6280\u672f\u89c4\u6a21': 'CRPS',

        'Croplands (agricultural soils)_\u6280\u672f\u6e17\u900f\u7387': 'CRPP',

        'Rice cultivation_\u6280\u672f\u89c4\u6a21': 'RICS',

        'Rice cultivation_\u6280\u672f\u6e17\u900f\u7387': 'RICP',

        'Combustion of fossil fuels and biomass_\u6280\u672f\u89c4\u6a21': 'COMS',

        'Combustion of fossil fuels and biomass_\u6280\u672f\u6e17\u900f\u7387': 'COMP',

        'Natural gas and oil systems_\u6280\u672f\u89c4\u6a21': 'GASS',

        'Natural gas and oil systems_\u6280\u672f\u6e17\u900f\u7387': 'GASP',

        'Coal mining activities_\u6280\u672f\u89c4\u6a21': 'CLMS',

        'Coal mining activities_\u6280\u672f\u6e17\u900f\u7387': 'CLMP',

        'Waste_\u6280\u672f\u89c4\u6a21': 'WSTS',

        'Waste_\u6280\u672f\u6e17\u900f\u7387': 'WSTP',

        'Industrial Processes_\u6280\u672f\u89c4\u6a21': 'INDS',

        'Industrial Processes_\u6280\u672f\u6e17\u900f\u7387': 'INDP',

    }

    df = df.rename(columns=col_map)

    eps = 1e-9

    if '\u7164\u70ad\u6d88\u8d39\u91cf' in df.columns:

        df['CCI'] = df['\u7164\u70ad\u6d88\u8d39\u91cf'] / (df['GDP'] + eps)

        print('  CCI was constructed successfully')

    else:

        print('  CCI could not be constructed because the coal consumption column is missing')

    if '\u77f3\u6cb9\u6d88\u8d39\u91cf' in df.columns and '\u5929\u7136\u6c14\u6d88\u8d39\u91cf' in df.columns:

        df['OAG'] = df['\u77f3\u6cb9\u6d88\u8d39\u91cf'] + df['\u5929\u7136\u6c14\u6d88\u8d39\u91cf']

        print('  OAG was constructed successfully')

    elif '\u77f3\u6cb9\u6d88\u8d39\u91cf' in df.columns:

        df['OAG'] = df['\u77f3\u6cb9\u6d88\u8d39\u91cf']

        print('  OAG was constructed successfully using oil consumption only')

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

                print(f'  {target} was replaced by its residual with respect to [Year, PGDP, URB]')

            else:

                print(f'  {target} is missing; residualization skipped')

    else:

        missing_controls = [c for c in control_vars if c not in df.columns]

        print(f'  Missing control variables {missing_controls}; residualization skipped')

    if 'POV' in df.columns and 'GDP' in df.columns:

        df['POVP'] = df['POV'] / (df['GDP'] + eps)

        print('  POVP was constructed successfully')

    if 'LOV' in df.columns and 'GDP' in df.columns:

        df['LOVP'] = df['LOV'] / (df['GDP'] + eps)

        print('  LOVP was constructed successfully')

    if 'IVA' in df.columns and 'GDP' in df.columns:

        df['IOVP'] = df['IVA'] / (df['GDP'] + eps)

        print('  IOVP was constructed successfully')

    ch4_features = [

        'GDP', 'PGDP', 'CCI', 'OAG', 'APEV',

        'POP', 'URB', 'DS',

        'POV', 'LOV', 'ALA', 'RPA', 'IVA', 'IOVP', 'POVP', 'LOVP',

        'LIVS', 'LIVP',

        'CRPS', 'CRPP',

        'RICS', 'RICP',

        'COMS', 'COMP',

        'GASS', 'GASP',

        'CLMS', 'CLMP',

        'WSTS', 'WSTP',

        'INDS', 'INDP',

        'POL',

        'TEMP', 'PRCP',

    ]

    feature_cols = [c for c in ch4_features if c in df.columns]

    missing = set(ch4_features) - set(feature_cols)

    if missing:

        print(f'  Unavailable features: {sorted(missing)}')

    agri_group = ['LOV', 'POV', 'ALA', 'APEV', 'RPA']

    agri_present = [c for c in agri_group if c in feature_cols]

    print('\n  --- Agricultural production group pre-screening ---')

    print(f'  Available variables in the group ({len(agri_present)}): {agri_present}')

    agri_removed = []

    protected_agri = {'LOV', 'RPA'}

    if len(agri_present) > 2:

        corr_matrix = df[agri_present].corr().abs()

        max_remove_count = 2

        while len(agri_removed) < max_remove_count:

            current_candidates = [c for c in agri_present if c not in protected_agri and c not in agri_removed]

            if not current_candidates:

                break

            sub_corr = corr_matrix.loc[current_candidates, [c for c in agri_present if c not in agri_removed]]

            np.fill_diagonal(sub_corr.values, 0)

            mean_corrs = sub_corr.mean(axis=1)

            worst_candidate = mean_corrs.idxmax()

            max_val = mean_corrs.max()

            if max_val > 0.4:

                agri_removed.append(worst_candidate)

                print(f'  Pre-screening exclusion: {worst_candidate} (within-group mean absolute correlation = {max_val:.3f})')

            else:

                break

        print(

            f'  Agricultural production group processed: LOV and RPA were protected; excluded {agri_removed}; retained {[c for c in agri_present if c not in agri_removed]}')

    else:

        print('  Agricultural group pre-screening was not applied because the number of variables was insufficient')

    feature_cols = [c for c in feature_cols if c not in agri_removed]

    print(f'  Number of features before global VIF screening ({len(feature_cols)}): {feature_cols}\n')

    X = df[feature_cols].copy()

    y = df['\u6392\u653e\u91cf'].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]

    if constant_cols:

        print(f'  Constant columns removed: {constant_cols}')

        X = X.drop(columns=constant_cols)

    removed_vif = []

    while True:

        vif_vals = [variance_inflation_factor(X.values.astype(float), i)

                    for i in range(X.shape[1])]

        vif_df = pd.DataFrame({'feature': X.columns, 'VIF': vif_vals})

        max_vif = vif_df['VIF'].max()

        if max_vif < 10:

            break

        high_vif_feats = set(vif_df[vif_df['VIF'] >= 10]['feature'])

        if 'POP' in high_vif_feats:

            worst = 'POP'

            curr_vif = vif_df.loc[vif_df['feature'] == 'POP', 'VIF'].values[0]

            print(f'  VIF priority removal: POP (VIF={curr_vif:.2f})')

        elif 'POV' in high_vif_feats:

            worst = 'POV'

            curr_vif = vif_df.loc[vif_df['feature'] == 'POV', 'VIF'].values[0]

            print(f'  VIF priority removal: POV (VIF={curr_vif:.2f})')

        else:

            worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']

            print(f'  VIF removal: {worst} (VIF={max_vif:.2f})')

        removed_vif.append(worst)

        X = X.drop(columns=[worst])

    keep = X.columns.tolist()

    print(f'  VIF screening completed: {len(keep)} features retained, {len(removed_vif)} features removed')

    if removed_vif:

        print(f'  VIF removal list: {removed_vif}')

    print(f'  Final features: {keep}')

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

    print('  Tuning Random Forest ...')

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

    print(f'    Best CV R2: {s.best_value:.4f}  params: {p}')

    return p, s.best_value

def tune_xgb(X, y, n_trials):

    print('  Tuning XGBoost ...')

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

    print(f'    Best CV R2: {s.best_value:.4f}  params: {p}')

    return p, s.best_value

def tune_lgb(X, y, n_trials):

    print('  Tuning LightGBM ...')

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

    print(f'    Best CV R2: {s.best_value:.4f}  params: {p}')

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

                     'R\xb2': r2_score(yte, pred),

                     'RMSE': np.sqrt(mean_squared_error(yte, pred)),

                     'MAE': mean_absolute_error(yte, pred)})

    fold_df = pd.DataFrame(rows)

    overall = {'Model': model_name,

               'R\xb2_mean': fold_df['R\xb2'].mean(), 'R\xb2_std': fold_df['R\xb2'].std(),

               'RMSE_mean': fold_df['RMSE'].mean(), 'MAE_mean': fold_df['MAE'].mean()}

    print(f"  {model_name:15s}  R\xb2={overall['R\xb2_mean']:.4f}\xb1{overall['R\xb2_std']:.4f}"

          f"  RMSE={overall['RMSE_mean']:.2f}  MAE={overall['MAE_mean']:.2f}")

    return fold_df, overall, np.array(all_true), np.array(all_pred)

def plot_model_comparison(metrics_list, out_dir):

    df_m = pd.DataFrame(metrics_list)

    models = df_m['Model'].tolist()

    colors = ['#2196F3', '#FF9800', '#4CAF50']

    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (col, label, fmt) in zip(axes, [

        ('R\xb2_mean', 'Mean R\xb2', '.4f'),

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

    print(f'  Saved to {p}')

def plot_fold_r2(fold_results, out_dir):

    colors = {'Random Forest': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}

    fig, ax = plt.subplots(figsize=(7, 4))

    for name, fd in fold_results.items():

        ax.plot(fd['Fold'], fd['R\xb2'], marker='o', linewidth=2,

                label=name, color=colors[name])

    ax.set_xlabel('Fold', fontsize=11)

    ax.set_ylabel('R\xb2', fontsize=11)

    ax.set_title(f'{GAS} \u2014 R\xb2 by Fold (Time-Series CV)', fontsize=12, fontweight='bold')

    ax.legend(fontsize=10)

    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    p = os.path.join(out_dir, 'fig2_fold_r2.png')

    plt.savefig(p, dpi=300, bbox_inches='tight')

    plt.close()

    print(f'  Saved to {p}')

def save_metrics_excel(metrics_list, fold_results, out_dir):

    p = os.path.join(out_dir, 'model_metrics_summary.xlsx')

    with pd.ExcelWriter(p, engine='openpyxl') as w:

        pd.DataFrame(metrics_list).to_excel(w, sheet_name='Overall', index=False)

        for name, fd in fold_results.items():

            fd.to_excel(w, sheet_name=name.replace(' ', '_')[:31], index=False)

    print(f'  Metrics table saved to {p}')

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

        shap.summary_plot(sv[:, idx], Xsc[top_f], plot_type='dot',

                          show=False, plot_size=None)

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

    print('  Global SHAP beeswarm, violin, bar plots, and data tables were saved for all models')

    print(f'  Summary table saved to {path_all}')

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

    print(f'  Saved to {p}')

    df_cmp.to_excel(os.path.join(out_dir, 'shap_comparison_table.xlsx'))

    return df_cmp

def plot_regional_shap(best_model, X_scaled, df_raw, out_dir, top_n=10):

    rdir = os.path.join(out_dir, 'regional_shap')

    os.makedirs(rdir, exist_ok=True)

    Xsc = X_scaled.copy()

    Xsc['_region'] = df_raw['region'].values

    top_n_actual = min(top_n, X_scaled.shape[1])

    long_rows = []

    regional_signed_matrix_data = []

    for region in sorted(Xsc['_region'].unique()):

        mask = Xsc['_region'] == region

        X_reg = Xsc.loc[mask, X_scaled.columns]

        if len(X_reg) < 2:

            print(f'  Region {region} has too few samples ({len(X_reg)}); skipped')

            continue

        bg_size = min(len(X_reg), 100)

        bg = shap.sample(X_reg, bg_size, random_state=42)

        print(f'  > Region {region}: using local background (sample size: {len(X_reg)})')

        explainer = shap.TreeExplainer(best_model, data=bg,

                                       feature_perturbation='interventional')

        try:

            sv = explainer.shap_values(X_reg, check_additivity=False)

            if isinstance(sv, list): sv = sv[0]

        except Exception as e:

            print(f'  Regional analysis failed for {region}: {e}')

            continue

        mean_signed = pd.Series(sv.mean(axis=0), index=X_reg.columns)

        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_reg.columns)

        total = mean_abs.sum()

        top_feats = mean_abs.nlargest(top_n_actual).index.tolist()[::-1]

        idx = [X_reg.columns.get_loc(f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(8, 5))

        shap.summary_plot(sv[:, idx], X_reg[top_feats], show=False, plot_size=None)

        ax = plt.gca()

        ax.set_title(f'{GAS} Region: {region} \u2014 SHAP (Local Background)',

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

        matrix_row = {'Region': region}

        for feat in X_scaled.columns:

            matrix_row[feat] = mean_signed.get(feat, 0.0)

        regional_signed_matrix_data.append(matrix_row)

    df_signed_matrix = pd.DataFrame(regional_signed_matrix_data)

    df_signed_matrix.set_index('Region', inplace=True)

    path_matrix = os.path.join(out_dir, 'regional_feature_mean_shap_matrix_SIGNED.xlsx')

    df_signed_matrix.to_excel(path_matrix)

    print(f'  Signed regional feature mean SHAP matrix saved to {os.path.basename(path_matrix)}')

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

    print(f'  Regional SHAP plots saved for {df_long["region"].nunique()} regions')

    print('  Long table saved to regional_shap_long.xlsx')

    print('  Wide table saved to regional_shap_wide.xlsx')

    print('  Region-specific tables saved to regional_shap_by_region.xlsx')

    return df_long

if __name__ == '__main__':

    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, df_raw, feat_cols = prepare_data(FILE)

    print('\n' + '=' * 60)

    print(f'Step 2: Optuna hyperparameter tuning ({N_TRIALS} trials per model)')

    rf_params, rf_r2 = tune_rf(X, y, N_TRIALS)

    xgb_params, xgb_r2 = tune_xgb(X, y, N_TRIALS)

    lgb_params, lgb_r2 = tune_lgb(X, y, N_TRIALS)

    print('\n' + '=' * 60)

    print('Step 3: 5-fold time-series cross-validation')

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

    print('Step 4: Save model performance outputs')

    plot_model_comparison(metrics_list, OUT_DIR)

    plot_fold_r2(fold_results, OUT_DIR)

    save_metrics_excel(metrics_list, fold_results, OUT_DIR)

    print('\n' + '=' * 60)

    print('Step 5: Full-sample training and SHAP analysis')

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

        r2 = next(m['R\xb2_mean'] for m in metrics_list if m['Model'] == name)

        if r2 > best_r2_val:

            best_r2_val, best_model, best_name = r2, model, name

    print(f'  Best model: {best_name}  (CV R2={best_r2_val:.4f})')

    print('\n' + '=' * 60)

    print('Step 6: Global SHAP plots for all models')

    plot_global_shap(shap_dict, Xsc_dict, OUT_DIR)

    plot_shap_comparison(shap_dict, Xsc_dict, OUT_DIR)

    if 'region' in df_raw.columns:

        print('\n' + '=' * 60)

        print('Step 7: Regional SHAP analysis')

        plot_regional_shap(best_model, Xsc_all, df_raw, OUT_DIR)
