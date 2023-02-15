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
    print(len(x))
    j = 0
    while j < len(x):
        x_mat[j][n] = (x[j] - meanValue) / stdValue
        if x_mat[j][n] > 0:
            print(x_mat[j][n])
        j = j + 1
    print("The ", n, "feature  is normal.")


"-------------------------------------读取文件划分数据集-----------------------------------------"
source_file = open("../dataset/KDDCUP99/processed/kddcup.data_10_percent_corrected.csv")
handled_file = open("../dataset/KDDCUP99/processed/10_percent_zscore0206.csv", 'w', newline='')
lines = source_file.readlines()
line_nums = len(lines)
print(line_nums)

# 创建line_nums行 para_num列的矩阵
x_mat = np.zeros((line_nums, 42))

# 划分数据集
for i in range(line_nums):
    line = lines[i].strip()
    item_mat = line.split(',')
    x_mat[i, :] = item_mat[0:42]    # 获取42个特征
source_file.close()
print(x_mat.shape)

"--------------------------------获取某列特征并依次标准化并赋值-----------------------------"
print(len(x_mat[:, 0]))     # 获取某列数据 number=494021
print(len(x_mat[0, :]))     # 获取某行数据 number=42

# 标准化处理
# TCP连接基本特征
ZscoreNormalization(x_mat[:, 0], 0)    # duration
ZscoreNormalization(x_mat[:, 4], 4)    # src_bytes
ZscoreNormalization(x_mat[:, 5], 5)    # dst_bytes
ZscoreNormalization(x_mat[:, 7], 7)    # wrong_fragment
ZscoreNormalization(x_mat[:, 8], 8)    # urgent

# TCP连接内容特征
ZscoreNormalization(x_mat[:, 9], 9)    # hot
ZscoreNormalization(x_mat[:, 10], 10)  # num_failed_logins
ZscoreNormalization(x_mat[:, 12], 12)  # num_compromised
# ZscoreNormalization(x_mat[:, 14], 14)  # su_attempted
ZscoreNormalization(x_mat[:, 15], 15)  # num_root
ZscoreNormalization(x_mat[:, 16], 16)  # num_file_creations
ZscoreNormalization(x_mat[:, 17], 17)  # num_shells
ZscoreNormalization(x_mat[:, 18], 18)  # num_access_files
# ZscoreNormalization(x_mat[:, 19], 19)  # num_outbound_cmds

# 基于时间的网络流量统计特征
ZscoreNormalization(x_mat[:, 22], 22)  # count
ZscoreNormalization(x_mat[:, 23], 23)  # srv_count
ZscoreNormalization(x_mat[:, 24], 24)  # serror_rate
ZscoreNormalization(x_mat[:, 25], 25)  # srv_serror_rate
ZscoreNormalization(x_mat[:, 26], 26)  # rerror_rate
ZscoreNormalization(x_mat[:, 27], 27)  # srv_rerror_rate
ZscoreNormalization(x_mat[:, 28], 28)  # same_srv_rate
ZscoreNormalization(x_mat[:, 29], 29)  # diff_srv_rate
ZscoreNormalization(x_mat[:, 30], 30)  # srv_diff_host_rate

# 基于主机的网络流量统计特征
ZscoreNormalization(x_mat[:, 31], 31)  # dst_host_count
ZscoreNormalization(x_mat[:, 32], 32)  # dst_host_srv_count
ZscoreNormalization(x_mat[:, 33], 33)  # dst_host_same_srv_rate
ZscoreNormalization(x_mat[:, 34], 34)  # dst_host_diff_srv_rate
ZscoreNormalization(x_mat[:, 35], 35)  # dst_host_same_src_port_rate
ZscoreNormalization(x_mat[:, 36], 36)  # dst_host_srv_diff_host_rate
ZscoreNormalization(x_mat[:, 37], 37)  # dst_host_serror_rate
ZscoreNormalization(x_mat[:, 38], 38)  # dst_host_srv_serror_rate
ZscoreNormalization(x_mat[:, 39], 39)  # dst_host_rerror_rate
ZscoreNormalization(x_mat[:, 40], 40)  # dst_host_srv_rerror_rate

# 文件写入操作
csv_writer = csv.writer(handled_file)
i = 0
while i < len(x_mat[:, 0]):
    csv_writer.writerow(x_mat[i, :])
    i = i + 1
handled_file.close()
