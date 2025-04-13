"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/12 21:37
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   tmp.py
**************************************
"""
# import pandas as pd
#
# # 读取原始CSV文件
# df = pd.read_csv(r'D:\06_engineering\03_analysis\pj_gptp_simulation\old_version\gptp_time_errors_improved.csv')  # 请将'your_file.csv'替换为你的实际文件名
#
# # 选择需要的列
# selected_columns = ['Time (s)', 'Hop 0', 'Hop 1', 'Hop 9', 'Hop 24', 'Hop 49', 'Hop 74', 'Hop 99']
# new_df = df[selected_columns]
#
# # 保存到新的CSV文件
# new_df.to_csv('selected_hops.csv', index=False)
#
# print("新的CSV文件已保存为 'selected_hops.csv'")

import csv


def filter_csv(input_file, output_file, start_row=2, step=4):
    """
    过滤CSV文件，保留特定间隔的行

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        start_row: 开始保留的行号(从1开始计数)
        step: 保留的行间隔
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入标题行(第一行)
        header = next(reader)
        writer.writerow(header)

        # 处理数据行
        for i, row in enumerate(reader, 1):  # 从1开始计数
            if (i + 1 - start_row) % step == 0:  # 判断是否是需要保留的行
                writer.writerow(row)


if __name__ == "__main__":
    input_filename = r"/old_version/main_test6/selected_hops.csv"  # 输入文件名
    output_filename = "output.csv"  # 输出文件名

    filter_csv(input_filename, output_filename)

    print(f"处理完成，结果已保存到 {output_filename}")

