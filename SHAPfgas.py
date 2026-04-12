# ===================================================================
#                          Libraries & Settings
# ===================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import matplotlib.font_manager as fm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import optuna

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# ===================================================================
#                          Function Definitions
# ===================================================================

def prepare_data():
    """Load data, perform feature engineering, and filter by VIF."""
    print("--- Loading and preparing data ---")
    df = pd.read_excel("FGAS_shap.xlsx")

    # Feature Engineering
    eps = 1e-6
    df['Industry_GDP_Ratio'] = df['工业产值'] / (df['GDP'] + eps)

    # Log transformation for skewed variables
    log_features = ['GDP']
    for col in log_features:
        df[col] = np.log(df[col] + eps)

    target_column = '排放量'
    feature_columns = [col for col in df.columns if col not in ['Year', 'country', 'region', target_column]]
    df = df.sort_values(by='Year')

    X = df[feature_columns]
    y = df[target_column]

    # Handle missing values and constants
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # Multicollinearity check using Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Keep features with VIF < 10
    low_vif_features = vif_data[vif_data['VIF'] < 10]['feature'].tolist()
    print(f"--- VIF filtering complete. Retained {len(low_vif_features)} features. ---")
    return X[low_vif_features], y, df, low_vif_features


def run_full_shap_analysis(model_params, output_dir_name, X, y, df, feature_columns):
    """Train model and execute SHAP global and regional analysis."""
    print(f"\n🚀 Starting analysis for '{output_dir_name}'...")
    os.makedirs(output_dir_name, exist_ok=True)

    # Scaling features to [0, 1] for better SHAP interpretability
    final_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled_all = X.copy()
    X_scaled_all[feature_columns] = final_scaler.fit_transform(X[feature_columns])

    # Fit final model
    final_model = xgb.XGBRegressor(**model_params)
    final_model.fit(X_scaled_all, y)

    # --- Global SHAP Analysis ---
    explainer_all = shap.TreeExplainer(final_model)
    shap_values_all = explainer_all.shap_values(X_scaled_all)

    # Global Beeswarm Plot
    plt.figure()
    shap.summary_plot(shap_values_all, X_scaled_all, max_display=10, show=False)
    plt.title('Global SHAP Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/global_shap_summary_beeswarm.png", dpi=300)
    plt.close()

    # Global Bar Plot (Feature Importance)
    plt.figure()
    shap.summary_plot(shap_values_all, X_scaled_all, plot_type="bar", max_display=10, show=False)
    plt.title('Global Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/global_feature_importance_bar.png", dpi=300)
    plt.close()

    # --- Regional SHAP Analysis ---
    X_scaled_all['region'] = df['region'].values
    regions = X_scaled_all['region'].unique()
    all_regional_shap_list = []

    # Sample global background for interventional perturbation
    global_background = shap.sample(X_scaled_all[feature_columns], 100, random_state=42)

    for region in regions:
        region_mask = X_scaled_all['region'] == region
        X_region = X_scaled_all.loc[region_mask, feature_columns]
        if X_region.empty: continue

        # Use global background for USA vs global trend comparison
        background_data = X_region if region != 'USA' else global_background
        explainer_reg = shap.TreeExplainer(final_model, data=background_data, feature_perturbation="interventional")
        shap_values_region = explainer_reg.shap_values(X_region)

        # Regional Summary Plots
        plt.figure()
        shap.summary_plot(shap_values_region, X_region, show=False)
        plt.title(f'Regional SHAP: {region}')
        plt.tight_layout()
        plt.savefig(f"{output_dir_name}/regional_shap_{region}.png", dpi=300)
        plt.close()

        # Aggregating mean absolute SHAP values for export
        region_shap_df = pd.DataFrame(np.abs(shap_values_region), columns=X_region.columns).mean().reset_index()
        region_shap_df.columns = ['feature', 'mean_abs_shap']
        region_shap_df['region'] = region
        all_regional_shap_list.append(region_shap_df)

    if all_regional_shap_list:
        df_all_regions = pd.concat(all_regional_shap_list, ignore_index=True)
        df_all_regions.to_excel(f"{output_dir_name}/regional_shap_summary_all.xlsx", index=False)

    print(f"✅ Analysis for '{output_dir_name}' completed.")


# ===================================================================
#                           Main Execution
# ===================================================================
if __name__ == "__main__":

    X, y, df, feature_columns = prepare_data()

    # --- Hyperparameter Optimization via Optuna ---
    print("\n--- Starting Optuna search ---")


    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }

        # TimeSeriesSplit for valid temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        r2_scores_list = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            r2_scores_list.append(r2_score(y_test, model.predict(X_test)))

        return np.mean(r2_scores_list)


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # --- Compare Optimized vs Manual Parameters ---
    optuna_params = study.best_params
    optuna_params['random_state'] = 42

    manual_params = {
        'n_estimators': 782, 'learning_rate': 0.169, 'max_depth': 6,
        'subsample': 0.730, 'colsample_bytree': 0.746, 'reg_alpha': 0.280,
        'reg_lambda': 0.319, 'min_child_weight': 2, 'gamma': 0.443,
        'random_state': 42
    }

    # Run Analysis 1: Manual Config
    run_full_shap_analysis(
        model_params=manual_params,
        output_dir_name="Results_Manual_Params",
        X=X, y=y, df=df, feature_columns=feature_columns
    )

    # Run Analysis 2: Optuna Optimized Config
    run_full_shap_analysis(
        model_params=optuna_params,
        output_dir_name="Results_Optuna_Optimized",
        X=X, y=y, df=df, feature_columns=feature_columns
    )

    print("\n🎉 Pipeline execution finished successfully.")