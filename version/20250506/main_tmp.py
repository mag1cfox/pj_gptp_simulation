import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

# 设置全局字体和字号
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'mathtext.fontset': 'stix'
})

# 1. 定义文件路径
case0_path = r"D:\06_engineering\03_analysis\pj_gptp_simulation\version\20250506\output_data_text_v7\case1_no_domain_data.csv"
case1_path = r"D:\06_engineering\03_analysis\pj_gptp_simulation\version\20250506\output_data_text_v7\case3_data.csv"
case2_path = r"D:\06_engineering\03_analysis\pj_gptp_simulation\version\20250506\output_data_text_v7\case1_data.csv"
case3_path = r"D:\06_engineering\03_analysis\pj_gptp_simulation\version\20250506\output_data_text_v7\case2_data.csv"

# 2. 创建输出目录
output_dir = "output_cases_comparison_v7"
os.makedirs(output_dir, exist_ok=True)
print(f"数据将保存到: {output_dir}")

# 3. 数据预处理函数
def preprocess_data(data_col):
    """处理数据列，去除NaN值并转换为numpy数组"""
    data = data_col.values
    data = np.nan_to_num(data, nan=np.nanmean(data))
    return data

# 4. 读取并处理数据
try:
    # 读取Case0数据 (最后一列)
    df0 = pd.read_csv(case0_path)
    print("\nCase0文件加载成功！前5行示例：")
    print(df0.head())
    data0 = preprocess_data(df0.iloc[:, -1])  # 获取最后一列数据
    print(f"处理Case0数据: 有效数据点={len(data0)}")

    # 读取Case1数据
    df1 = pd.read_csv(case1_path)
    print("\nCase1文件加载成功！前5行示例：")
    print(df1.head())
    data1 = preprocess_data(df1.iloc[:, -1])  # 获取最后一列数据
    print(f"处理Case1数据: 有效数据点={len(data1)}")

    # 读取Case2数据
    df2 = pd.read_csv(case2_path)
    print("\nCase2文件加载成功！前5行示例：")
    print(df2.head())
    data2 = preprocess_data(df2.iloc[:, -1])  # 获取最后一列数据
    print(f"处理Case2数据: 有效数据点={len(data2)}")

    # 读取Case3数据
    df3 = pd.read_csv(case3_path)
    print("\nCase3文件加载成功！前5行示例：")
    print(df3.head())
    data3 = preprocess_data(df3.iloc[:, -1])  # 获取最后一列数据
    print(f"处理Case3数据: 有效数据点={len(data3)}")

except Exception as e:
    print(f"\n数据文件加载失败: {e}")
    raise

# 5. 创建结果DataFrame并保存
result_df = pd.DataFrame({
    "Case0_no_domain": data0,
    "Case1": data1,
    "Case2": data2,
    "Case3": data3
})

output_csv_path = os.path.join(output_dir, "cases_comparison_data.csv")
result_df.to_csv(output_csv_path, index=False)
print(f"\n所有数据已合并保存到: {output_csv_path}")

# 6. 绘制主图和插入式放大图
plt.figure(figsize=(15, 8))
ax_main = plt.gca()

# 定义线条样式
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']  # 蓝,红,绿,紫
linestyles = ['-', '--', ':', '-.']
linewidths = [2.5, 2.5, 2.5, 2.5]
labels = ['Case0 (no domain)', 'Case1', 'Case2', 'Case3']

# 定义绘制顺序 (Case1, Case2, Case3, Case0)
plot_order = [ 1,2, 3, 0]  # 对应data1, data2, data3, data0

# 绘制主图 - 按新顺序绘制
x = np.arange(len(data0))
for i in plot_order:
    data = [data0, data1, data2, data3][i]
    ax_main.plot(x, data,
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                label=labels[i],
                zorder=4-i)  # zorder确保图层顺序

# 设置主图属性 (保持不变)
ax_main.set_xlabel('Test Time (s)', fontweight='bold')
ax_main.set_ylabel('Time Error (ns)', fontweight='bold')
ax_main.set_title('Time Error Comparison Across Different Cases', fontweight='bold')
ax_main.grid(True, which="both", linestyle='--', color='gray', alpha=0.7)
legend = ax_main.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                      frameon=True, framealpha=0.8, edgecolor='black')

# 定义放大区域 (保持不变)
zoom_x_start = 1500
zoom_x_end = 2500
zoom_y_min = -500
zoom_y_max = 500
rect = Rectangle((zoom_x_start, zoom_y_min),
                 zoom_x_end - zoom_x_start,
                 zoom_y_max - zoom_y_min,
                 linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
ax_main.add_patch(rect)

# 创建插入式放大图 (放在右下角)
ax_zoom = ax_main.inset_axes([0.6, 0.1, 0.35, 0.35])

# 绘制放大图 - 按同样顺序绘制
for i in plot_order:
    data = [data0, data1, data2, data3][i]
    ax_zoom.plot(x[zoom_x_start:zoom_x_end], data[zoom_x_start:zoom_x_end],
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                zorder=4-i)

# 设置放大图属性 (保持不变)
ax_zoom.set_xlim(zoom_x_start, zoom_x_end)
ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
ax_zoom.grid(True, which="both", linestyle='--', color='gray', alpha=0.7)
ax_zoom.set_title('Zoom: x[1500-2500], y[-300-300]', fontsize=14, pad=10)
ax_zoom.indicate_inset_zoom(ax_main, edgecolor="black")

# 调整布局以留出足够空间
plt.tight_layout()

# 保存图表 (保持不变)
output_plot_path = os.path.join(output_dir, "cases_comparison_with_inset.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"折线图(带插入式放大图)已保存到: {output_plot_path}")
