"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/21 13:56
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   Fig_32hops_TE.py
**************************************
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',  # 设置字体加粗
    'mathtext.fontset': 'stix'
})

# 读取Excel文件中的数据
filename = r'input_data\TE.xlsx'  # Excel文件名
data = pd.read_excel(filename, header=None, skiprows=1, usecols=[0, 1, 2])  # 跳过第一行，读取前三列

# 提取三列数据
one_hop_data = data.iloc[:, 2].values
seven_hop_data = data.iloc[:, 1].values
thirty_two_hop_data = data.iloc[:, 0].values

# 定义与前一个图表相同的颜色
colors = [
    '#ff7f0e',  # 深蓝色 (1 hop)
    '#A2142F',  # 深红色 (7 hops)
    '#0072BD'   # 灰色 (32 hops)
]

# 定义不同的线型和标记
linestyles = ['-', '-.', '--']
markers = ['o', '*', 'x']
markevery = 500  # 控制标记的密度，每500个数据点显示一个标记

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图，使用相同的颜色和线型风格
plt.plot(one_hop_data, color=colors[0], linestyle=linestyles[0], marker=markers[0],
         markevery=markevery, label='1 hop')
plt.plot(seven_hop_data, color=colors[1], linestyle=linestyles[1], marker=markers[1],
         markevery=markevery, label='7 hops')
plt.plot(thirty_two_hop_data, color=colors[2], linestyle=linestyles[2], marker=markers[2],
         markevery=markevery, label='32 hops', linewidth=2)  # 略微加粗32 hops线条

# 设置x轴范围
plt.xlim(1, 10800)

# 添加图例、标题和轴标签
plt.xlabel('Test Time(s)', color='black', fontweight='bold')
plt.ylabel('Time Error(ns)', color='black', fontweight='bold')

# 显示图例
plt.legend(fontsize=16)

# 设置坐标轴颜色为黑色
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# 设置背景为白色
ax.set_facecolor('white')

# 显示灰色网格虚线
plt.grid(True, linestyle='--', color='gray', alpha=0.7)

# 保存图形（可选）
plt.savefig(r'output_image\time_error_plot_v2.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()