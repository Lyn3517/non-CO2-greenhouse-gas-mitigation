import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')
FILE = 'X'
GAS = 'FGAS'
OUT_DIR = 'X'
FIXED_RF_PARAMS = {'n_estimators': 398, 'max_depth': 8, 'min_samples_leaf': 1, 'max_features': 0.387117813189669, 'random_state': 42, 'n_jobs': -1}
FIXED_XGB_PARAMS = {'n_estimators': 328, 'learning_rate': 0.011492047484731186, 'max_depth': 8, 'subsample': 0.736046266569358, 'colsample_bytree': 0.5104684695495867, 'reg_alpha': 7.601451943710624e-06, 'reg_lambda': 5.438194655460511e-06, 'random_state': 42, 'verbosity': 0}
FIXED_LGB_PARAMS = {'n_estimators': 700, 'learning_rate': 0.08873780198962208, 'max_depth': 6, 'num_leaves': 35, 'subsample': 0.8053278421084017, 'colsample_bytree': 0.5605826836476965, 'reg_alpha': 1.2335017858882492e-07, 'reg_lambda': 4.417718546044945, 'random_state': 42, 'verbose': -1}
DIM_COLORS = {'Economy': '#2196F3', 'Society': '#4CAF50', 'Industry': '#FF9800', 'Technology': '#9C27B0', 'Policy': '#F44336', 'Environment': '#00BCD4'}
FEATURE_DIM = {'GDP': 'Economy', 'PGDP': 'Economy', 'CCI': 'Economy', 'OAGCI': 'Economy', 'POP': 'Society', 'URB': 'Society', 'DS': 'Society', 'IVA': 'Industry', 'IOVP': 'Industry', 'INDS': 'Technology', 'INDP': 'Technology', 'POL': 'Policy', 'TEMP': 'Environment', 'PRCP': 'Environment'}

def winsorize_series(s, lower_pct=0.05, upper_pct=0.95):
    if s.isna().all() or s.nunique() <= 1:
        return s
    return s.clip(lower=s.quantile(lower_pct), upper=s.quantile(upper_pct))

def check_constant(s):
    if s.isna().all() or s.nunique() <= 1:
        return True
    return bool(s.std() < 1e-05 or s.max() - s.min() < 0.0001)

def fill_group_median(df, group_col, feat):
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    if group_col:
        df[feat] = df.groupby(group_col)[feat].transform(lambda x: x.fillna(x.median() if not x.isna().all() else df[feat].median()))

def quantize_to_signal(s, lower_pct, upper_pct):
    if s.isna().all() or s.nunique() <= 1:
        return pd.Series(0, index=s.index)
    q_low = s.quantile(lower_pct)
    q_high = s.quantile(upper_pct)
    signal = pd.Series(0, index=s.index)
    signal[s < q_low] = -1
    signal[s > q_high] = 1
    return signal

def prepare_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values('Year').reset_index(drop=True)
    col_map = {'\u6c14\u6e29': 'TEMP', '\u964d\u6c34': 'PRCP', '\u4eba\u53e3': 'POP', 'GDP': 'GDP', '\u4eba\u5747GDP': 'PGDP', '\u57ce\u9547\u5316\u7387': 'URB', '\u996e\u98df\u7ed3\u6784': 'DS', '\u653f\u7b56\u6570\u91cf': 'POL', '\u5de5\u4e1a\u4ea7\u503c': 'IVA', 'Industrial Processes_\u6280\u672f\u89c4\u6a21': 'INDS', 'Industrial Processes_\u6280\u672f\u6e17\u900f\u7387': 'INDP'}
    df = df.rename(columns=col_map)
    country_col = None
    for possible_col in ['Country', 'country', 'region']:
        if possible_col in df.columns:
            country_col = possible_col
            break
    if not country_col:
        raise ValueError('\u6570\u636e\u4e2d\u672a\u627e\u5230\u6709\u6548\u7684\u56fd\u5bb6/\u533a\u57df\u5217\uff0c\u65e0\u6cd5\u6267\u884c\u7ec4\u5185\u5904\u7406\uff01')
    pre_fill_cols = ['DS', 'TEMP', 'PRCP', 'Year', 'PGDP', 'URB', 'GDP', '\u7164\u70ad\u6d88\u8d39\u91cf', '\u77f3\u6cb9\u6d88\u8d39\u91cf', '\u5929\u7136\u6c14\u6d88\u8d39\u91cf', 'IVA']
    for c in pre_fill_cols:
        if c in df.columns:
            fill_group_median(df, country_col, c)
    eps = 1e-09
    if '\u7164\u70ad\u6d88\u8d39\u91cf' in df.columns and 'GDP' in df.columns:
        df['CCI'] = df['\u7164\u70ad\u6d88\u8d39\u91cf'] / (df['GDP'] + eps)
    oil_cols = [c for c in ['\u77f3\u6cb9\u6d88\u8d39\u91cf', '\u5929\u7136\u6c14\u6d88\u8d39\u91cf'] if c in df.columns]
    if oil_cols and 'GDP' in df.columns:
        df['OAGCI'] = df[oil_cols].sum(axis=1) / (df['GDP'] + eps)
    if 'IVA' in df.columns and 'GDP' in df.columns:
        df['IOVP'] = df['IVA'] / (df['GDP'] + eps)
    for c in ['CCI', 'OAGCI', 'IOVP']:
        if c in df.columns:
            fill_group_median(df, country_col, c)
    for target in ['TEMP', 'PRCP']:
        if target in df.columns:
            try:
                transformed = df.groupby(country_col)[target].transform(lambda x: (x - x.mean()) / (x.std() + eps) if x.std() > 0 else x - x.mean())
                transformed = winsorize_series(transformed, 0.05, 0.95)
                signal = quantize_to_signal(transformed, 0.05, 0.95)
                if check_constant(signal):
                    demean = df.groupby(country_col, group_keys=False).apply(lambda x: x[target] - x[target].mean())
                    signal = quantize_to_signal(demean, 0.2, 0.8)
                df[target] = signal
            except Exception as e:
                demean = df.groupby(country_col, group_keys=False).apply(lambda x: x[target] - x[target].mean())
                df[target] = quantize_to_signal(demean, 0.2, 0.8)
    if 'DS' in df.columns:
        try:
            residual_list = []
            control_vars = ['Year', 'PGDP', 'URB']
            for country, group in df.groupby(country_col):
                if len(group) >= 5:
                    X_g = group[control_vars].fillna(0)
                    y_g = group['DS']
                    lr = LinearRegression().fit(X_g, y_g)
                    res_g = y_g - lr.predict(X_g)
                else:
                    res_g = group['DS'] - group['DS'].mean()
                residual_list.append(res_g)
            ds_res = pd.concat(residual_list).sort_index()
            ds_res = winsorize_series(ds_res, 0.05, 0.95)
            df['_temp_res'] = ds_res
            ds_diff = df.groupby(country_col)['_temp_res'].diff().fillna(0)
            ds_diff = winsorize_series(ds_diff, 0.05, 0.95)
            ds_signal = quantize_to_signal(ds_diff, 0.2, 0.8)
            if check_constant(ds_signal):
                raw_diff = df.groupby(country_col)['DS'].diff().fillna(0)
                ds_signal = quantize_to_signal(raw_diff, 0.2, 0.8)
            df['DS'] = ds_signal
        except Exception as e:
            raw_diff = df.groupby(country_col)['DS'].diff().fillna(0)
            df['DS'] = quantize_to_signal(raw_diff, 0.2, 0.8)
        finally:
            if '_temp_res' in df.columns:
                df = df.drop(columns=['_temp_res'])
    exclude_features_by_const = []
    for feat in ['DS', 'TEMP', 'PRCP']:
        if feat in df.columns:
            if check_constant(df[feat]):
                exclude_features_by_const.append(feat)
    fgas_features = ['GDP', 'PGDP', 'CCI', 'OAGCI', 'POP', 'URB', 'DS', 'IVA', 'IOVP', 'INDS', 'INDP', 'POL', 'TEMP', 'PRCP']
    feature_cols = [c for c in fgas_features if c in df.columns and c not in exclude_features_by_const]
    X = df[feature_cols].copy()
    y = df['\u6392\u653e\u91cf'].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    while True:
        vif_vals = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({'feature': X.columns, 'VIF': vif_vals})
        max_vif = vif_df['VIF'].max()
        if max_vif < 10:
            break
        worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        X = X.drop(columns=[worst])
    keep = X.columns.tolist()
    return (X[keep], y, df, keep)

def evaluate_model(model, X, y, model_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = MinMaxScaler()
    rows, all_pred, all_true = ([], [], [])
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        Xtr = pd.DataFrame(scaler.fit_transform(X.iloc[tr]), columns=X.columns)
        Xte = pd.DataFrame(scaler.transform(X.iloc[te]), columns=X.columns)
        ytr, yte = (y.iloc[tr], y.iloc[te])
        m = model.__class__(**model.get_params())
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        all_pred.extend(pred)
        all_true.extend(yte.values)
        rows.append({'Fold': fold, 'R\xb2': r2_score(yte, pred), 'RMSE': np.sqrt(mean_squared_error(yte, pred)), 'MAE': mean_absolute_error(yte, pred)})
    fold_df = pd.DataFrame(rows)
    overall = {'Model': model_name, 'R\xb2_mean': fold_df['R\xb2'].mean(), 'R\xb2_std': fold_df['R\xb2'].std(), 'RMSE_mean': fold_df['RMSE'].mean(), 'MAE_mean': fold_df['MAE'].mean()}
    return (fold_df, overall, np.array(all_true), np.array(all_pred))

def plot_model_comparison(metrics_list, out_dir):
    df_m = pd.DataFrame(metrics_list)
    models = df_m['Model'].tolist()
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, label, fmt) in zip(axes, [('R\xb2_mean', 'Mean R\xb2', '.4f'), ('RMSE_mean', 'Mean RMSE', '.2f'), ('MAE_mean', 'Mean MAE', '.2f')]):
        vals = df_m[col].values
        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01, f'{v:{fmt}}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
    plt.suptitle(f'{GAS} \u2014 Model Performance Comparison', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_r2(fold_results, out_dir):
    colors = {'Random Forest': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, fd in fold_results.items():
        ax.plot(fd['Fold'], fd['R\xb2'], marker='o', linewidth=2, label=name, color=colors[name])
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('R\xb2', fontsize=11)
    ax.set_title(f'{GAS} \u2014 R\xb2 by Fold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_fold_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_excel(metrics_list, fold_results, out_dir):
    p = os.path.join(out_dir, 'model_metrics_summary.xlsx')
    with pd.ExcelWriter(p, engine='openpyxl') as w:
        pd.DataFrame(metrics_list).to_excel(w, sheet_name='Overall', index=False)
        for name, fd in fold_results.items():
            fd.to_excel(w, sheet_name=name.replace(' ', '_')[:31], index=False)

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
        ax.set_title(f'{name} \u2014 Global SHAP Beeswarm', fontsize=12, fontweight='bold')
        patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
        ax.legend(handles=patches, title='Dimension', fontsize=8, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3_beeswarm_{safe}.png'), dpi=300)
        plt.close()
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv[:, idx], Xsc[top_f], plot_type='violin', show=False, plot_size=None)
        ax = plt.gca()
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(lbl, 'Economy'), '#333'))
        ax.set_title(f'{name} \u2014 Global SHAP Violin', fontsize=12, fontweight='bold')
        ax.legend(handles=patches, title='Dimension', fontsize=8, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig3b_violin_{safe}.png'), dpi=300)
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_f, mean_abs[top_f], color=_dim_colors_for(top_f), edgecolor='white')
        ax.set_title(f'{name} \u2014 Feature Importance', fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=patches, title='Dimension', fontsize=8, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig4_bar_{safe}.png'), dpi=300)
        plt.close()
        total_shap = mean_abs.sum()
        df_table = pd.DataFrame({'model': name, 'feature': mean_abs.index, 'mean_abs_shap': mean_abs.values, 'shap_ratio': mean_abs.values / total_shap, 'dimension': [FEATURE_DIM.get(f, 'Unknown') for f in mean_abs.index]}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        df_table['rank'] = df_table.index + 1
        all_global_rows.append(df_table)
        df_table.to_excel(os.path.join(out_dir, f'global_shap_table_{safe}.xlsx'), index=False)
    path_all = os.path.join(out_dir, 'global_shap_table_all_models.xlsx')
    with pd.ExcelWriter(path_all, engine='openpyxl') as w:
        for df_t in all_global_rows:
            sheet_name = df_t['model'].iloc[0].replace(' ', '_')[:31]
            df_t.to_excel(w, sheet_name=sheet_name, index=False)

def plot_shap_comparison(shap_dict, X_sc_dict, out_dir, top_n=15):
    base_name = 'XGBoost' if 'XGBoost' in shap_dict else list(shap_dict.keys())[0]
    n_feats = len(X_sc_dict[base_name].columns)
    top_n_actual = min(top_n, n_feats)
    base_feats = pd.Series(np.abs(shap_dict[base_name]).mean(axis=0), index=X_sc_dict[base_name].columns).nlargest(top_n_actual).index.tolist()
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
        ax.barh(x + (i - 1) * w, vals, height=w, label=name, color=clr.get(name, '#999'), edgecolor='white', alpha=0.9)
    ax.set_yticks(x)
    ax.set_yticklabels(base_feats, fontsize=9)
    for tick, feat in zip(ax.get_yticklabels(), base_feats):
        tick.set_color(DIM_COLORS.get(FEATURE_DIM.get(feat, 'Economy'), '#333'))
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Comparison Across Models', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    dim_patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
    mdl_patches = [mpatches.Patch(color=clr[n], label=n) for n in clr]
    ax.legend(handles=dim_patches + mdl_patches, fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'fig5_shap_3model_comparison.png'), dpi=300)
    plt.close()
    df_cmp.to_excel(os.path.join(out_dir, 'shap_comparison_table.xlsx'))
    return df_cmp

def plot_regional_shap(best_model, X_scaled, df_raw, out_dir, top_n=10):
    rdir = os.path.join(out_dir, 'regional_shap')
    os.makedirs(rdir, exist_ok=True)
    Xsc = X_scaled.copy()
    region_col = 'region' if 'region' in df_raw.columns else 'Country' if 'Country' in df_raw.columns else 'country'
    Xsc['_region'] = df_raw[region_col].values
    top_n_actual = min(top_n, X_scaled.shape[1])
    long_rows = []
    matrix_rows = []
    path_by_region = os.path.join(out_dir, 'regional_shap_by_region.xlsx')
    excel_writer = pd.ExcelWriter(path_by_region, engine='openpyxl')
    for region in sorted(Xsc['_region'].unique()):
        mask = Xsc['_region'] == region
        X_reg = Xsc.loc[mask, X_scaled.columns]
        if len(X_reg) < 2:
            continue
        bg = shap.sample(X_reg, min(len(X_reg), 100), random_state=42)
        explainer = shap.TreeExplainer(best_model, data=bg, feature_perturbation='interventional')
        try:
            sv = explainer.shap_values(X_reg, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[0]
        except Exception as e:
            continue
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_reg.columns)
        mean_signed = pd.Series(sv.mean(axis=0), index=X_reg.columns)
        total = mean_abs.sum()
        region_matrix_dict = {'region': region}
        for feat in X_scaled.columns:
            region_matrix_dict[feat] = mean_signed.get(feat, 0.0)
        matrix_rows.append(region_matrix_dict)
        region_rows = []
        for feat in X_scaled.columns:
            val_abs = mean_abs.get(feat, 0.0)
            ratio = val_abs / total if total > 0 else 0
            dim = FEATURE_DIM.get(feat, 'Unknown')
            row_data = {'region': region, 'feature': feat, 'mean_abs_shap': val_abs, 'shap_ratio': ratio, 'dimension': dim}
            long_rows.append(row_data)
            region_rows.append(row_data)
        df_region = pd.DataFrame(region_rows).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        safe_sheet_name = str(region).replace(' ', '_')[:31]
        df_region.to_excel(excel_writer, sheet_name=safe_sheet_name, index=False)
        top_feats = mean_abs.nlargest(top_n_actual).index.tolist()[::-1]
        idx = [X_reg.columns.get_loc(f) for f in top_feats]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv[:, idx], X_reg[top_feats], show=False, plot_size=None)
        ax = plt.gca()
        ax.set_title(f'Region: {region} \u2014 SHAP', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(rdir, f'regional_{region}.png'), dpi=200)
        plt.close()
    excel_writer.close()
    df_long = pd.DataFrame(long_rows).sort_values(['region', 'mean_abs_shap'], ascending=[True, False])
    df_long.to_excel(os.path.join(out_dir, 'regional_shap_long.xlsx'), index=False)
    df_wide = df_long.pivot(index='region', columns='feature', values='mean_abs_shap')
    df_wide.to_excel(os.path.join(out_dir, 'regional_shap_wide.xlsx'))
    df_matrix = pd.DataFrame(matrix_rows).set_index('region')
    path_matrix = os.path.join(out_dir, 'regional_shap_mean_matrix.xlsx')
    df_matrix.to_excel(path_matrix)
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, df_raw, feat_cols = prepare_data(FILE)
    models = {'Random Forest': RandomForestRegressor(**FIXED_RF_PARAMS), 'XGBoost': xgb.XGBRegressor(**FIXED_XGB_PARAMS), 'LightGBM': lgb.LGBMRegressor(**FIXED_LGB_PARAMS)}
    metrics_list, fold_results = ([], {})
    for name, model in models.items():
        fd, ov, true_vals, pred_vals = evaluate_model(model, X, y, name)
        metrics_list.append(ov)
        fold_results[name] = fd
        pd.DataFrame({'y_true': true_vals, 'y_pred': pred_vals}).to_excel(os.path.join(OUT_DIR, f"cv_predictions_{name.replace(' ', '_')}.xlsx"), index=False)
    plot_model_comparison(metrics_list, OUT_DIR)
    plot_fold_r2(fold_results, OUT_DIR)
    save_metrics_excel(metrics_list, fold_results, OUT_DIR)
    scaler = MinMaxScaler()
    Xsc_all = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    shap_dict, Xsc_dict = ({}, {})
    best_model, best_r2_val, best_name = (None, -np.inf, '')
    for name, model in models.items():
        model.fit(Xsc_all, y)
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(Xsc_all)
        if isinstance(sv, list):
            sv = sv[0]
        shap_dict[name] = sv
        Xsc_dict[name] = Xsc_all.copy()
        r2 = next((m['R\xb2_mean'] for m in metrics_list if m['Model'] == name))
        if r2 > best_r2_val:
            best_r2_val, best_model, best_name = (r2, model, name)
    plot_global_shap(shap_dict, Xsc_dict, OUT_DIR, top_n=15)
    plot_shap_comparison(shap_dict, Xsc_dict, OUT_DIR, top_n=15)
    plot_regional_shap(best_model, Xsc_all, df_raw, OUT_DIR, top_n=10)
