"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/9 15:55
*  @Project :   pj_gptp_simulation
*  @Description :   对最大te绘制条形图
*  @FileName:   tiaoxingtu20250409.py
**************************************
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 数据准备
data = {
    'hops': [10, 25, 50, 75, 100],
    'interval=31.25ms': [0.543, 1.197, 2.331, 3.783, 5.083],
    'interval=125ms': [1.327, 1.568, 2.908, 4.299, 5.426],
    'interval=1s': [1.116, 3.858, 2.396, 3.617, 14.706]
}

df = pd.DataFrame(data)

# 设置学术期刊风格的配色方案
# 使用Nature期刊推荐的Okabe-Ito配色方案
colors = ['#E69F00', '#56B4E9', '#009E73']  # 橙色、蓝色、绿色

# 创建图表
plt.figure(figsize=(10, 6), dpi=300)  # 高分辨率适合出版物

# 设置字体为Times New Roman，学术论文常用
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 条形图参数
bar_width = 0.25  # 每个条形的宽度
x = np.arange(len(df['hops']))  # x轴位置

# 绘制条形图
bars = []
for i, col in enumerate(df.columns[1:]):
    bars.append(plt.bar(x + i*bar_width, df[col], width=bar_width,
                       color=colors[i], edgecolor='black', linewidth=0.5,
                       label=col))

# 添加图表元素
plt.xlabel('Number of Hops', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.title('Performance by Number of Hops and Interval', fontsize=14, fontweight='bold')
plt.xticks(x + bar_width, df['hops'], fontsize=11)
plt.yticks(fontsize=11)

# 添加图例
plt.legend(fontsize=11, frameon=True, shadow=False, edgecolor='black')

# 添加网格线（学术图表常用浅色网格）
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 保存图表（可根据需要选择格式）
# plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
# plt.savefig('bar_chart.pdf', dpi=300, bbox_inches='tight')
