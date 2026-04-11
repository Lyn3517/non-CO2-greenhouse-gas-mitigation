import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# ===== 读取并整理数据 =====
file_path = "数据.csv"  # 请修改为您的文件路径
df = pd.read_excel(file_path, header=None)

# 第一行是年份，第二行是数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# ===== 读取并整理数据 =====
file_path = "数据.csv"  # 请修改为您的文件路径
df = pd.read_excel(file_path, header=None)

# 第一行是年份，第二行是数据
years = df.iloc[0].values
years = pd.to_numeric(years, errors='coerce')
years = years[~np.isnan(years)]
years = years.astype(int)

emissions = df.iloc[1].values[:len(years)]
global_Gt = emissions / 1e6

# ===== 创建颜色映射：从蓝色到红色渐变 =====
cmap = cm.coolwarm  # 蓝色到红色的渐变色
norm = Normalize(vmin=float(np.nanmin(global_Gt)), vmax=float(np.nanmax(global_Gt)))

# ===== 绘图 =====
fig, ax1 = plt.subplots(figsize=(12, 7))

bar_width = 0.8
bar_edge_color = "#555555"

# 生成每个年份对应的颜色，并为每根柱子绘制渐变
for i, (year, val) in enumerate(zip(years, global_Gt)):
    # 获取每根柱子的基础颜色：根据值从蓝色到红色的渐变
    base_color = cmap(norm(val))

    # 创建渐变色：从白色到顶部颜色的渐变
    gradient = np.linspace(1, 0, 256).reshape(1, -1)  # 从白色到该柱子的颜色的渐变
    gradient = np.vstack((gradient, gradient))

    # 为每个柱子绘制渐变效果
    ax1.imshow(gradient, aspect="auto", cmap=cm.Blues, extent=[year - bar_width / 2, year + bar_width / 2, 0, val])

    # 绘制柱形，使用基础颜色
    ax1.bar(year, val, width=bar_width, color=base_color, edgecolor=None, linewidth=0)

# 顶部数值（紧贴柱形）
label_offset_ratio = 0.005  # 调整为适合位置
for year, val in zip(years, global_Gt):
    if not np.isnan(val):
        ax1.text(year, val * (1 + label_offset_ratio), f"{val:.1f}",
                 ha="center", va="bottom", fontsize=9, color="#333333")

# 平滑趋势线（EWMA）
smoothed = pd.Series(global_Gt).ewm(span=5, adjust=False).mean()
offset_ratio = 0.08  # 偏移比例增加趋势线位置
offset = float(np.nanmax(global_Gt)) * offset_ratio
ax1.plot(years, smoothed + offset, color="#d62728", linewidth=2.5, label="Smoothed trend")

# 设置坐标轴
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel(r"CO$_2$ emissions (GtCO$_2$-eq)", fontsize=12)
ymax = float(np.nanmax(global_Gt)) * 1.30 + offset
ax1.set_ylim(0, ymax)
ax1.set_xlim(years[0] - bar_width * 1.2, years[-1] + bar_width * 1.2)
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45, ha="right")

# 边框设置
ax1.grid(False)
ax1.spines['top'].set_visible(True)
ax1.spines['top'].set_color(bar_edge_color)
ax1.spines['right'].set_color(bar_edge_color)
ax1.spines['left'].set_color(bar_edge_color)
ax1.spines['bottom'].set_color(bar_edge_color)

# 图例
leg = ax1.legend(loc="upper left", frameon=True, fontsize=10)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_alpha(0.95)

plt.tight_layout()
output_path = "output_plot_final.png"  # 修改为保存路径
plt.savefig(output_path, dpi=200)
plt.show()
