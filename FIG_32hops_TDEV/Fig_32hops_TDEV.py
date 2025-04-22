"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/21 13:49
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   Fig_32hops_TDEV.py
**************************************
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体和字号
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',  # 设置字体加粗
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'mathtext.fontset': 'stix'
})

# 读取Excel文件中的数据
filename = r'input_data\TDEV.xlsx'  # 替换为您的Excel文件名

try:
    # 尝试方法1：跳过第一行（如果是标题行）
    df = pd.read_excel(filename, header=None, skiprows=1)
    data = df.values.astype(float)
except ValueError:
    try:
        # 尝试方法2：使用pandas数值解析功能
        df = pd.read_excel(filename, header=None)
        # 只保留全数值行
        numeric_rows = []
        for i, row in df.iterrows():
            try:
                numeric_row = [float(x) for x in row]
                numeric_rows.append(numeric_row)
            except (ValueError, TypeError):
                print(f"跳过非数值行: {row}")

        if not numeric_rows:
            raise ValueError("没有找到可用的数值数据")

        data = np.array(numeric_rows)
    except:
        # 尝试方法3：让用户指定具体的数据范围
        print("无法自动解析数据，尝试指定具体数据范围")
        # 可以修改为您知道的确切数据范围，例如从A2:I10
        df = pd.read_excel(filename, header=None, usecols=range(9), skiprows=1)
        data = df.values.astype(float)

print(f"成功读取数据，形状为: {data.shape}")

# 分离数据
xData = data[:, 0]  # 第一列是x轴数据
yData1 = data[:, 1]  # 第二列是第一组y轴数据
yData2 = data[:, 2]  # 第三列是第二组y轴数据
yData3 = data[:, 3]  # 第四列是第三组y轴数据
yData4 = data[:, 4]  # 第五列是第四组y轴数据
yData5 = data[:, 5]  # 第六列是第五组y轴数据
yData6 = data[:, 6]  # 第七列是第六组y轴数据
yData7 = data[:, 7]  # 第八列是第七组y轴数据
yData8 = data[:, 8]  # 第九列是第八组y轴数据

# 创建图形窗口
fig, ax = plt.subplots(figsize=(8, 6))

# 设置为对数刻度
ax.set_xscale('log')
ax.set_yscale('log')

# 定义更具区分度的颜色、线型和标记
colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
marker_sizes = [8, 8, 10, 10, 8, 10, 12, 10]  # 标记大小
markevery = 5  # 每5个数据点显示一个标记

# 绘制数据
ax.plot(xData, yData1, color=colors[0], linestyle=linestyles[0], marker=markers[0],
        markersize=marker_sizes[0], markevery=markevery, label='1 hop')
ax.plot(xData, yData2, color=colors[1], linestyle=linestyles[1], marker=markers[1],
        markersize=marker_sizes[1], markevery=markevery, label='2 hops')
ax.plot(xData, yData3, color=colors[2], linestyle=linestyles[2], marker=markers[2],
        markersize=marker_sizes[2], markevery=markevery, label='3 hops')
ax.plot(xData, yData4, color=colors[3], linestyle=linestyles[3], marker=markers[3],
        markersize=marker_sizes[3], markevery=markevery, label='4 hops')
ax.plot(xData, yData5, color=colors[4], linestyle=linestyles[4], marker=markers[4],
        markersize=marker_sizes[4], markevery=markevery, label='5 hops')
ax.plot(xData, yData6, color=colors[5], linestyle=linestyles[5], marker=markers[5],
        markersize=marker_sizes[5], markevery=markevery, label='6 hops')
ax.plot(xData, yData7, color=colors[6], linestyle=linestyles[6], marker=markers[6],
        markersize=marker_sizes[6], markevery=markevery, label='7 hops')
ax.plot(xData, yData8, color=colors[7], linestyle=linestyles[7], marker=markers[7],
        markersize=marker_sizes[7], markevery=markevery, label='32 hops')

# 添加图例
ax.legend()

# 添加轴标签
ax.set_xlabel('τ(s)')
ax.set_ylabel('TDEV(s)')

# 显示网格
ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)

# 调整图形布局
plt.tight_layout()

# 保存图形
plt.savefig(r'output_image\TDEV_32hops_v3.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()