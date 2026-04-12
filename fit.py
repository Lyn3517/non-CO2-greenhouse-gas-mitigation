import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from scipy.spatial import KDTree

# Plotting Configuration
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def scale_point_size(y_true_log_values, base_size=36, high_thresh=0.9, low_ratio=0.5):
    """Dynamic scaling of scatter points to reduce visual crowding in high-density areas."""
    if len(y_true_log_values) == 0:
        return np.array([])
    upper_limit = np.quantile(y_true_log_values, high_thresh)
    sizes = np.full_like(y_true_log_values, base_size, dtype=float)
    sizes[y_true_log_values >= upper_limit] = base_size * low_ratio
    return sizes


def plot_single_model_on_ax(ax, x_final, y_final, sizes, metrics, model_name, gas_label, color, band_tolerance=0.5):
    """Plots individual model results with 1:1 line and performance metrics."""
    r2_orig, mae_orig, rmse_orig = metrics
    min_val, max_val = min(x_final.min(), y_final.min()), max(x_final.max(), y_final.max())
    padding = (max_val - min_val if max_val > min_val else 1.0) * 0.05
    lim_lower, lim_upper = min_val - padding, max_val + padding

    # Draw 1:1 line and tolerance band
    x_band = np.array([lim_lower, lim_upper])
    ax.fill_between(x_band, x_band - band_tolerance, x_band + band_tolerance, color='#dddddd', alpha=0.6, zorder=0)
    ax.plot([lim_lower, lim_upper], [lim_lower, lim_upper], 'k--', lw=1, zorder=1)

    # Scatter plot with scaled sizes
    ax.scatter(x_final, y_final, s=sizes, color=color, edgecolor='black', linewidth=0.5, alpha=0.8, zorder=2)

    ax.set_xlim(lim_lower, lim_upper)
    ax.set_ylim(lim_lower, lim_upper)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Predicted Values (logₑ)", fontsize=9, weight='bold')
    ax.set_ylabel("True Values (logₑ)", fontsize=9, weight='bold')
    ax.set_title(model_name, fontsize=11, weight='bold', pad=6)

    # Display Metrics
    metrics_text = f"R²: {r2_orig:.3f}\nMAE: {mae_orig:.2f}\nRMSE: {rmse_orig:.2f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9, color='#D20000', weight='bold', va='top')


def _preprocess_and_sample_enhanced(y_true, y_pred, n_backbone=30, n_flesh=350, max_zero_points=6):
    """
    Advanced sampling strategy:
    1. Log1p transformation for skewed emission data.
    2. Backbone sampling via KDTree to ensure range coverage.
    3. Flesh sampling (binned) to handle high-density regions.
    4. Jitter added to prevent overlap.
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true >= 0) & (y_pred >= 0)
    y_t, y_p = y_true[valid], y_pred[valid]
    if len(y_t) < 2: return None

    metrics = (r2_score(y_t, y_p), mean_absolute_error(y_t, y_p), np.sqrt(mean_squared_error(y_t, y_p)))
    y_t_log, y_p_log = np.log1p(y_t), np.log1p(y_p)

    # Sampling Logic
    indices = np.arange(len(y_t_log))
    targets = np.linspace(y_t_log.min(), y_t_log.max(), n_backbone)
    _, backbone_idx = KDTree(y_t_log[:, None]).query(targets[:, None], k=1)

    available = np.setdiff1d(indices, backbone_idx)
    flesh_idx = resample(available, n_samples=min(n_flesh, len(available)), random_state=42)

    final_idx = np.unique(np.concatenate([backbone_idx.flatten(), flesh_idx]))

    # Add Jitter for visualization
    np.random.seed(42)
    x_final = y_p_log[final_idx] + np.random.uniform(-0.02, 0.02, len(final_idx))
    y_final = y_t_log[final_idx] + np.random.uniform(-0.02, 0.02, len(final_idx))

    return x_final, y_final, scale_point_size(y_t_log[final_idx]), metrics


def load_gas_model_data(file_path, gas_label):
    sheets = pd.read_excel(file_path, sheet_name=None)
    models = {name: (df.iloc[:, 0].values, df.iloc[:, 1].values) for name, df in sheets.items() if df.shape[1] >= 2}
    return {'gas_label': gas_label, 'models': models}


def plot_multi_gas_multi_model_layout(all_gas_data, save_path):
    colors = {r'CH$_4$': '#89a9d1', r'N$_2$O': '#ffb866', r'F-Gases': '#bbd36f'}
    target_models = ['LightGBM', 'XGBoost', 'RandomForest']

    fig = plt.figure(figsize=(len(all_gas_data) * 3.5, len(target_models) * 3.5), dpi=200)
    gs = GridSpec(len(target_models), len(all_gas_data), figure=fig, wspace=0.25, hspace=0.38)

    for i, gas_data in enumerate(all_gas_data):
        label = gas_data['gas_label']
        model_results = []
        for m_name in target_models:
            processed = _preprocess_and_sample_enhanced(*gas_data['models'][m_name]) if m_name in gas_data[
                'models'] else None
            r2 = processed[3][0] if processed else -np.inf
            model_results.append((m_name, processed, r2))

        # Sort models by R2 within each gas column
        model_results.sort(key=lambda x: x[2], reverse=True)

        for j, (m_name, processed, _) in enumerate(model_results):
            ax = fig.add_subplot(gs[j, i])
            if j == 0:
                ax.text(0.5, 1.25, label, ha='center', va='bottom', fontsize=15, weight='bold', transform=ax.transAxes)

            if processed:
                plot_single_model_on_ax(ax, *processed, m_name, label, colors.get(label, '#8064A2'))
            else:
                ax.text(0.5, 0.5, "Missing Data", ha='center', va='center')

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    file_mapping = {'CH4.xlsx': r'CH$_4$', 'N2O.xlsx': r'N$_2$O', 'FGAS.xlsx': r'F-Gases'}
    data_list = [load_gas_model_data(f, l) for f, l in file_mapping.items() if os.path.exists(f)]
    plot_multi_gas_multi_model_layout(data_list, "Model_Performance_Comparison.png")