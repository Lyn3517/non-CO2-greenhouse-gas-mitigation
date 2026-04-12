import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from adjustText import adjust_text
from matplotlib.colors import ListedColormap, BoundaryNorm

# Configuration
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# Paths

data_file = "groupinfo.xlsx"
region_mapping_file = "地区中英文对照表.xlsx"

output_dendrogram = os.path.join(data_dir, 'dendrogram_no_title_top.tif')
output_pca_scatter = os.path.join(data_dir, '1128PCA2.png')
output_heatmap = os.path.join(data_dir, 'heatmap_no_title_top.tif')

# Data Loading
data = pd.read_excel(os.path.join(data_dir, data_file), sheet_name='CO2-2')
mapping_df = pd.read_excel(os.path.join(data_dir, region_mapping_file))
zh_to_en_region = dict(zip(mapping_df['地区'], mapping_df['region']))
data['region'] = data['region'].map(zh_to_en_region).fillna(data['region'])

# Feature Engineering
features = ['CH4年均排放量', 'N2O年均排放量', 'F年均排放量']
categorical_features = [
    '达峰数量', 'CH4-量和强度', 'F-量和强度', 'F第一排放源',
    'CH4第一排放源', 'N2O第一排放源', 'N2O-量和强度'
]
data_encoded = pd.get_dummies(data[categorical_features], drop_first=True)
X = pd.concat([data[features], data_encoded], axis=1)

# Standardization & PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# Loading Analysis
loadings = pca.components_[:2].T
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=X.columns)

print("--- PC1 Top Contributors ---")
print(loadings_df.sort_values(by='PC1', key=abs, ascending=False)['PC1'].head(5))
print("\n--- PC2 Top Contributors ---")
print(loadings_df.sort_values(by='PC2', key=abs, ascending=False)['PC2'].head(5))

# Hierarchical Clustering
Z = linkage(X_pca, method='ward')
best_n_clusters = 3
final_clusters = fcluster(Z, best_n_clusters, criterion='maxclust')
data['Hierarchical_Cluster'] = final_clusters

# Plot: Dendrogram
plt.figure(figsize=(18, 12))
ax_dendro = plt.gca()
dendrogram(Z, labels=data['region'].values, leaf_rotation=90, ax=ax_dendro)
ax_dendro.xaxis.tick_top()
ax_dendro.xaxis.set_label_position('top')
plt.xlabel('Region', labelpad=20)
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(output_dendrogram, dpi=500, bbox_inches='tight')

# Plot: PCA Scatter
plt.figure(figsize=(12, 8))
ax = plt.gca()

cluster_colors_hex = {1: '#1F117B', 2: '#E35183', 3: '#F4E500'}
cmap_custom = ListedColormap(list(cluster_colors_hex.values()))
bounds = np.arange(1, len(cluster_colors_hex) + 2)
norm = BoundaryNorm(bounds, cmap_custom.N)

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=data['Hierarchical_Cluster'],
                     cmap=cmap_custom,
                     norm=norm,
                     s=100, alpha=0.8, zorder=1)

cbar = plt.colorbar(scatter, ticks=np.arange(1.5, len(cluster_colors_hex) + 1.5))
cbar.set_ticklabels(list(cluster_colors_hex.keys()))

texts = [ax.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=10, fontweight='bold')
         for i, txt in enumerate(data['region'])]
adjust_text(texts, x=X_pca[:, 0], y=X_pca[:, 1], add_objects=[scatter],
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.5, shrinkA=5))

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xlabel('Principal Component I', labelpad=20)
plt.ylabel('Principal Component II')
plt.grid(False)
plt.tight_layout()
plt.savefig(output_pca_scatter, dpi=500, bbox_inches='tight', transparent=True)

# Plot: Cluster Heatmap
cluster_centers = []
for cluster_id in range(1, best_n_clusters + 1):
    idx = np.where(data['Hierarchical_Cluster'] == cluster_id)[0]
    if len(idx) > 0:
        cluster_centers.append(X_scaled[idx].mean(axis=0))

if len(cluster_centers) > 0:
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=list(X.columns),
                                      index=[f'Cluster {i+1}' for i in range(len(cluster_centers))])
    plt.figure(figsize=(14, 8))
    ax_heatmap = plt.gca()
    sns.heatmap(cluster_centers_df, cmap='YlOrRd', center=0, ax=ax_heatmap)
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_label_position('top')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_heatmap, dpi=500, bbox_inches='tight')

# Metrics
dist_original = squareform(pdist(X_scaled, metric='euclidean'))
dist_pca = squareform(pdist(X_pca, metric='euclidean'))
correlation = np.corrcoef(dist_original.ravel(), dist_pca.ravel())[0, 1]
print(f"\nDistance Correlation (Original vs PCA): {correlation:.4f}")