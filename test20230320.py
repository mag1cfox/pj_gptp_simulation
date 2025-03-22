"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/3/20 11:25
*  @Project :   pj_gptp_simulation
*  @Description :读取Excel文件中指定列的数据，计算绝对值的最大值、最小值和均值，并将结果输出到新的CSV文件中
*  @FileName:   test20230320.py
**************************************
"""
import pandas as pd
import numpy as np


def process_csv_columns(input_file, output_file, columns_to_process):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 创建一个结果DataFrame用于存储统计结果
    results = pd.DataFrame(columns=['Column', 'Max_Absolute', 'Min_Absolute', 'Mean_Absolute'])

    # 处理每一个指定的列
    for col_idx in columns_to_process:
        # 检查列索引是否存在
        if col_idx < len(df.columns):
            # 获取列名
            col_name = df.columns[col_idx]

            # 获取列数据
            column_data = df.iloc[:, col_idx]

            # 计算绝对值
            abs_values = column_data.abs()

            # 计算统计量
            max_abs = abs_values.max()
            min_abs = abs_values.min()
            mean_abs = abs_values.mean()

            # 添加到结果DataFrame
            new_row = pd.DataFrame({
                'Column': [f'Column_{col_idx + 1}'],  # 列编号从1开始
                'Max_Absolute': [max_abs],
                'Min_Absolute': [min_abs],
                'Mean_Absolute': [mean_abs]
            })
            results = pd.concat([results, new_row], ignore_index=True)
        else:
            print(f"警告: 列索引 {col_idx} 超出范围，已跳过")

    # 保存结果到CSV文件
    results.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")


# 指定要处理的列索引（注意：CSV列从1开始，但Python索引从0开始）
columns_to_process = [9, 24, 49, 74, 99]  # 对应CSV中的第10、25、50、75、100列

# 调用函数处理数据
input_file = r"D:\06_engineering\03_analysis\pj_gptp_simulation\data2.csv"  # 替换为您的CSV文件路径
output_file = "statistics_results3.csv"  # 输出的CSV文件名
process_csv_columns(input_file, output_file, columns_to_process)