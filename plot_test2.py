import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体（Windows 自带）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建 pic 文件夹（如果不存在）
if not os.path.exists('pic'):
    os.makedirs('pic')

# 读取CSV文件，假设文件名为 'data.csv'
df = pd.read_csv(r'D:\06_engineering\03_analysis\pj_gptp_simulation\.vscode\data1.csv', header=None)

# 设置全局字体大小
plt.rcParams.update({'font.size': 18})

# 遍历每一列数据
for i in range(df.shape[1]):
    # 创建一个新的图形
    plt.figure(figsize=(3.2, 2.54))

    # 绘制折线图
    plt.plot(df[i], linewidth=1)

    # 设置x轴和y轴的标签
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('TE (μs)')

    # 设置x轴和y轴的范围
    plt.xlim(0, 3200)
    plt.ylim(-2, 2)

    # 设置y轴的刻度间隔
    plt.yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

    # 设置图名
    plt.title(f'{i + 1}跳TE结果')

    # 保存图形到 pic 文件夹下
    plt.savefig(f'pic/{i + 1}_跳TE结果.jpg', dpi=200, bbox_inches='tight')

    # 关闭图形，避免内存泄漏
    plt.close()