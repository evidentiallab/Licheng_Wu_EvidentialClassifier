# coding:utf-8
# Z-score标准化处理

import numpy as np
import pandas as pd
import csv

# 全局变量
global x_mat


# 数据标准化
def ZscoreNormalization(x, n):
    meanValue = np.mean(x)
    stdValue = np.std(x)
    # print(len(x))
    j = 0
    while j < len(x):
        x_mat[j][n] = (x[j] - meanValue) / stdValue
        if x_mat[j][n] > 0:
            print(x_mat[j][n])
        j = j + 1
    print("The ", n, "feature  is normal.")


# "--------------------------------获取某列特征并依次标准化并赋值-----------------------------"
# # print(len(x_mat[:, 0]))     # 获取某列数据 number=494021
# # print(len(x_mat[0, :]))     # 获取某行数据 number=42
#
# # 标准化处理
# # TCP连接基本特征
# ZscoreNormalization(x_mat[:, 0], 0)      # duration
# ZscoreNormalization(x_mat[:, 81], 81)    # src_bytes
# ZscoreNormalization(x_mat[:, 82], 82)    # dst_bytes
# ZscoreNormalization(x_mat[:, 85], 85)    # wrong_fragment
# ZscoreNormalization(x_mat[:, 86], 86)    # urgent
#
# # TCP连接内容特征
# ZscoreNormalization(x_mat[:, 87], 87)    # hot
# ZscoreNormalization(x_mat[:, 88], 88)  # num_failed_logins
# ZscoreNormalization(x_mat[:, 91], 91)  # num_compromised
# ZscoreNormalization(x_mat[:, 92], 92)  # root_shell
# ZscoreNormalization(x_mat[:, 93], 93)  # su_attempted
# ZscoreNormalization(x_mat[:, 94], 94)  # num_root
# ZscoreNormalization(x_mat[:, 95], 95)  # num_file_creations
# ZscoreNormalization(x_mat[:, 96], 96)  # num_shells
# ZscoreNormalization(x_mat[:, 97], 97)  # num_access_files
# ZscoreNormalization(x_mat[:, 98], 98)  # num_outbound_cmds
#
# # 基于时间的网络流量统计特征
# ZscoreNormalization(x_mat[:, 102], 102)  # count
# ZscoreNormalization(x_mat[:, 103], 103)  # srv_count
# ZscoreNormalization(x_mat[:, 104], 104)  # serror_rate
# ZscoreNormalization(x_mat[:, 105], 105)  # srv_serror_rate
# ZscoreNormalization(x_mat[:, 106], 106)  # rerror_rate
# ZscoreNormalization(x_mat[:, 107], 107)  # srv_rerror_rate
# ZscoreNormalization(x_mat[:, 108], 108)  # same_srv_rate
# ZscoreNormalization(x_mat[:, 109], 109)  # diff_srv_rate
# ZscoreNormalization(x_mat[:, 110], 110)  # srv_diff_host_rate
#
# # 基于主机的网络流量统计特征
# ZscoreNormalization(x_mat[:, 111], 111)  # dst_host_count
# ZscoreNormalization(x_mat[:, 112], 112)  # dst_host_srv_count
# ZscoreNormalization(x_mat[:, 113], 113)  # dst_host_same_srv_rate
# ZscoreNormalization(x_mat[:, 114], 114)  # dst_host_diff_srv_rate
# ZscoreNormalization(x_mat[:, 115], 115)  # dst_host_same_src_port_rate
# ZscoreNormalization(x_mat[:, 116], 116)  # dst_host_srv_diff_host_rate
# ZscoreNormalization(x_mat[:, 117], 117)  # dst_host_serror_rate
# ZscoreNormalization(x_mat[:, 118], 118)  # dst_host_srv_serror_rate
# ZscoreNormalization(x_mat[:, 119], 119)  # dst_host_rerror_rate
# ZscoreNormalization(x_mat[:, 120], 120)  # dst_host_srv_rerror_rate

if __name__ == '__main__':
    n_feature = 124
    if n_feature == 121:
        source_file = open('../dataset/KDDCUP99/processed/one-hot_121.csv')
        handled_file = open('../dataset/KDDCUP99/Z-score_121(dummy).csv', 'w', newline='')
        lines = source_file.readlines()
        line_nums = len(lines)
        # print(line_nums)
        # 创建line_nums行 para_num列的矩阵
        x_mat = np.zeros((line_nums, 144))
        # 划分数据集
        for i in range(line_nums):
            line = lines[i].strip()
            item_mat = line.split(',')
            x_mat[i, :] = item_mat[0:144]
        source_file.close()
        # print(x_mat.shape)
        for n in range(0,121):
            ZscoreNormalization(x_mat[:, n], n)
        # 文件写入操作
        csv_writer = csv.writer(handled_file)
        i = 0
        while i < len(x_mat[:, 0]):
            csv_writer.writerow(x_mat[i, :])
            i = i + 1
        handled_file.close()

    if n_feature == 124:
        source_file = open('../dataset/KDDCUP99/processed/one-hot_124.csv')
        handled_file = open('../dataset/KDDCUP99/Z-score_124(dummy).csv', 'w', newline='')
        lines = source_file.readlines()
        line_nums = len(lines)
        # print(line_nums)
        # 创建line_nums行 para_num列的矩阵
        x_mat = np.zeros((line_nums, 147))
        # 划分数据集
        for i in range(line_nums):
            line = lines[i].strip()
            item_mat = line.split(',')
            x_mat[i, :] = item_mat[0:147]
        source_file.close()
        # print(x_mat.shape)
        for n in range(0,124):
            ZscoreNormalization(x_mat[:, n], n)
        # 文件写入操作
        csv_writer = csv.writer(handled_file)
        i = 0
        while i < len(x_mat[:, 0]):
            csv_writer.writerow(x_mat[i, :])
            i = i + 1
        handled_file.close()
