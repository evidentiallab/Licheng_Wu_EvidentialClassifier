# coding:utf-8
# 归一化处理(Min-Max标准化)

import numpy as np
import pandas as pd
import csv

# 全局变量
global x_mat


# 数据归一化
def MinmaxNormalization(x, n):
    minValue = np.min(x)
    maxValue = np.max(x)
    print(minValue, maxValue)
    print(len(x))
    j = 0
    while j < len(x):
        x_mat[j][n] = (x[j] - minValue) / (maxValue - minValue)
        # if x_mat[j][n]>0:
        # print(x_mat[j][n])
        j = j + 1
    print("The ", n, "feature  is normal.")


if __name__ == '__main__':
    n_feature = 121

    if n_feature == 124:
        fr = open('../dataset/KDDCUP99/one-hot_124.csv')
        data_file = open('../dataset/KDDCUP99/Min-Max-only.csv', 'w', newline='')
        lines = fr.readlines()
        line_nums = len(lines)
        print(line_nums)
        # 创建line_nums行 para_num列的矩阵
        x_mat = np.zeros((line_nums, 147))

        # 划分数据集
        for i in range(line_nums):
            line = lines[i].strip()
            item_mat = line.split(',')
            x_mat[i, :] = item_mat[0:147]  # 获取42个特征
        fr.close()
        # print(x_mat.shape)
        # print(len(x_mat[:, 0]))   # 获取某列数据 494021
        # print(len(x_mat[0, :]))   # 获取某行数据 42
        # TCP连接基本特征
        MinmaxNormalization(x_mat[:, 0], 0)  # duration
        MinmaxNormalization(x_mat[:, 81], 81)  # src_bytes
        MinmaxNormalization(x_mat[:, 82], 82)  # dst_bytes
        MinmaxNormalization(x_mat[:, 85], 85)  # wrong_fragment
        MinmaxNormalization(x_mat[:, 86], 86)  # urgent
        # TCP连接内容特征
        MinmaxNormalization(x_mat[:, 87], 87)  # hot
        MinmaxNormalization(x_mat[:, 88], 88)  # num_failed_logins
        MinmaxNormalization(x_mat[:, 91], 91)  # num_compromised
        # MinmaxNormalization(x_mat[:, 92], 92)  # root_shell
        # MinmaxNormalization(x_mat[:, 93], 93)  # su_attempted
        MinmaxNormalization(x_mat[:, 97], 97)  # num_root
        MinmaxNormalization(x_mat[:, 98], 98)  # num_file_creations
        MinmaxNormalization(x_mat[:, 99], 99)  # num_shells
        MinmaxNormalization(x_mat[:, 100], 100)  # num_access_files
        MinmaxNormalization(x_mat[:, 101], 101)  # num_outbound_cmds
        # 基于时间的网络流量统计特征
        MinmaxNormalization(x_mat[:, 105], 105)  # count
        MinmaxNormalization(x_mat[:, 106], 106)  # srv_count
        MinmaxNormalization(x_mat[:, 107], 107)  # serror_rate
        MinmaxNormalization(x_mat[:, 108], 108)  # srv_serror_rate
        MinmaxNormalization(x_mat[:, 109], 109)  # rerror_rate
        MinmaxNormalization(x_mat[:, 110], 110)  # srv_rerror_rate
        MinmaxNormalization(x_mat[:, 111], 111)  # same_srv_rate
        MinmaxNormalization(x_mat[:, 112], 112)  # diff_srv_rate
        MinmaxNormalization(x_mat[:, 113], 113)  # srv_diff_host_rate
        # 基于主机的网络流量统计特征
        MinmaxNormalization(x_mat[:, 114], 114)  # dst_host_count
        MinmaxNormalization(x_mat[:, 115], 115)  # dst_host_srv_count
        MinmaxNormalization(x_mat[:, 116], 116)  # dst_host_same_srv_rate
        MinmaxNormalization(x_mat[:, 117], 117)  # dst_host_diff_srv_rate
        MinmaxNormalization(x_mat[:, 118], 118)  # dst_host_same_src_port_rate
        MinmaxNormalization(x_mat[:, 119], 119)  # dst_host_srv_diff_host_rate
        MinmaxNormalization(x_mat[:, 120], 120)  # dst_host_serror_rate
        MinmaxNormalization(x_mat[:, 121], 121)  # dst_host_srv_serror_rate
        MinmaxNormalization(x_mat[:, 122], 122)  # dst_host_rerror_rate
        MinmaxNormalization(x_mat[:, 123], 123)  # dst_host_srv_rerror_rate
        # 文件写入操作
        csv_writer = csv.writer(data_file)
        i = 0
        while i < len(x_mat[:, 0]):
            csv_writer.writerow(x_mat[i, :])
            i = i + 1
        data_file.close()

    elif n_feature == 121:
        fr = open('../dataset/KDDCUP99/processed/one-hot_121.csv')
        data_file = open('../dataset/KDDCUP99/feature_121.csv', 'w', newline='')
        lines = fr.readlines()
        line_nums = len(lines)
        print(line_nums)
        # 创建line_nums行 para_num列的矩阵
        x_mat = np.zeros((line_nums, 144))

        # 划分数据集
        for i in range(line_nums):
            line = lines[i].strip()
            item_mat = line.split(',')
            x_mat[i, :] = item_mat[0:144]  # 获取144列
        fr.close()

        for n in range(0,34):
            MinmaxNormalization(x_mat[:, n], n)

        csv_writer = csv.writer(data_file)
        i = 0
        while i < len(x_mat[:, 0]):
            csv_writer.writerow(x_mat[i, :])
            i = i + 1
        data_file.close()
