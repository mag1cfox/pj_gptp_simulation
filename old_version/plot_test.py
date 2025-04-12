"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/2/17 16:21
*  @Project :   pj_gptp_simulation
*  @Description :   尝试画计算出来的概率图
*  @FileName:   plot_test.py
**************************************
"""
import csv


def read_column_from_csv(file_path, column_index):
    column_data = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > column_index:  # 确保该行有足够的列
                try:
                    # 将字符串转换为浮点数，并保留三位小数
                    value = float(row[column_index])
                    column_data.append(round(value, 3))  # 保留三位小数
                except ValueError:
                    # 如果转换失败（例如非数字数据），跳过该值或记录错误
                    print(f"Warning: 无法将 '{row[column_index]}' 转换为浮点数，已跳过。")

    return column_data





if __name__ == '__main__':

    # 使用示例
    # file_path = r'D:\06_engineering\03_analysis\pj_gptp_simulation\.vscode\data5.csv'  # 替换为你的CSV文件路径
    file_path = r'/old_version/data5.csv'  # 替换为你的CSV文件路径
    column_index = 0  # 第10列的索引是9
    # column_data = read_column_from_csv(file_path, column_index)

    list=[100,51,29,24,20,24,38,51]
    column_data=[]
    for i in list:
        # print(i)
        # print(type(i))
        tmp = i -1
        column_data.append(read_column_from_csv(file_path, tmp))

    for l_tmp in column_data:
        # 初始化计数器
        count_0_2 = 0
        count_0_4 = 0
        count_0_6 = 0
        count_0_8 = 0
        count_1_0 = 0
        count_1_2 = 0
        count_1_4 = 0
        count_1_6 = 0
        count_1_8 = 0
        count_2_0 = 0
        for num in l_tmp:
            # 遍历列表
            if abs(num) <= 0.2:
                count_0_2 += 1
            if abs(num) <= 0.4:
                count_0_4 += 1
            if abs(num) <= 0.6:
                count_0_6 += 1
            if abs(num) <= 0.8:
                count_0_8 += 1
            if abs(num) <= 1.0:
                count_1_0 += 1
            if abs(num) <= 1.2:
                count_1_2 += 1
            if abs(num) <= 1.4:
                count_1_4 += 1
            if abs(num) <= 1.6:
                count_1_6 += 1
            if abs(num) <= 1.8:
                count_1_8 += 1
            if abs(num) <= 2.0:
                count_2_0 += 1
        print(f"list=[{count_0_2},{count_0_4},{count_0_6},{count_0_8},{count_1_0},{count_1_2},{count_1_4},{count_1_6},{count_1_8},{count_2_0}]")
        print(round((count_0_2/3198),3))
        print(round((count_0_4/3198),3))
        print(round((count_0_6/3198),3))
        print(round((count_0_8/3198),3))
        print(round((count_1_0/3198),3))
        print(round((count_1_2/3198),3))
        print(round((count_1_4/3198),3))
        print(round((count_1_6/3198),3))
        print(round((count_1_8/3198),3))
        print(round((count_2_0/3198),3))
        print("============")


