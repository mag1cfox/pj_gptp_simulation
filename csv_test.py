"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/2/17 9:52
*  @Project :   pj_gptp_simulation
*  @Description :   尝试将元组保存到csv中
*  @FileName:   csv_test.py
**************************************
"""

import csv
import os


def save_tuple_to_csv(tuple_data, filename='data.csv'):
    # 检查文件是否存在
    file_exists = os.path.isfile(filename)

    # 读取现有数据（如果文件存在）
    existing_data = []
    if file_exists:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_data = list(reader)

    # 找到第一个空列
    column_index = 0
    if existing_data:
        max_columns = max(len(row) for row in existing_data)
        for i in range(max_columns + 1):
            if all(i >= len(row) or row[i] == '' for row in existing_data):
                column_index = i
                break

    # 将 tuple 数据添加到正确的列
    for i, value in enumerate(tuple_data):
        row_index = i
        if row_index >= len(existing_data):
            existing_data.append([''] * (column_index + 1))
        while len(existing_data[row_index]) <= column_index:
            existing_data[row_index].append('')
        existing_data[row_index][column_index] = value

    # 写入数据到 CSV 文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(existing_data)


# 示例使用
my_tuple = (1, 2, 3, 4, 5)
save_tuple_to_csv(my_tuple)