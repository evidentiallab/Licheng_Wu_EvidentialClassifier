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


"-------------------------------------读取文件划分数据集-----------------------------------------"
fr = open("../dataset/KDDCUP99/processed/10_percent_zscore0206.csv")
data_file = open("../dataset/KDDCUP99/processed/10_percent_normal0206.csv",'w',newline='')
lines = fr.readlines()
line_nums = len(lines)
print(line_nums)


# 创建line_nums行 para_num列的矩阵
x_mat = np.zeros((line_nums, 42))

# 划分数据集
for i in range(line_nums):
    line = lines[i].strip()
    item_mat = line.split(',')
    x_mat[i, :] = item_mat[0:42]    #获取42个特征
fr.close()
print(x_mat.shape)

"--------------------------------获取某列特征并依次标准化并赋值-----------------------------"
print(len(x_mat[:, 0])) #获取某列数据 494021
print(len(x_mat[0, :])) #获取某行数据 42

#归一化处理
MinmaxNormalization(x_mat[:, 0], 0)    #duration
MinmaxNormalization(x_mat[:, 4], 4)    #src_bytes
MinmaxNormalization(x_mat[:, 5], 5)    #dst_bytes
MinmaxNormalization(x_mat[:, 7], 7)    #wrong_fragment
MinmaxNormalization(x_mat[:, 8], 8)    #urgent

MinmaxNormalization(x_mat[:, 9], 9)    #hot
MinmaxNormalization(x_mat[:, 10], 10)  #num_failed_logins
MinmaxNormalization(x_mat[:, 12], 12)  #num_compromised
# MinmaxNormalization(x_mat[:, 14], 14)  #su_attempte
MinmaxNormalization(x_mat[:, 15], 15)  #num_root
MinmaxNormalization(x_mat[:, 16], 16)  #num_file_creations
MinmaxNormalization(x_mat[:, 17], 17)  #num_shells
MinmaxNormalization(x_mat[:, 18], 18)  #num_access_files
# MinmaxNormalization(x_mat[:, 19], 19)  #num_outbound_cmds

MinmaxNormalization(x_mat[:, 22], 22)  #count
MinmaxNormalization(x_mat[:, 23], 23)  #srv_count
MinmaxNormalization(x_mat[:, 24], 24)  #serror_rate
MinmaxNormalization(x_mat[:, 25], 25)  #srv_serror_rate
MinmaxNormalization(x_mat[:, 26], 26)  #rerror_rate
MinmaxNormalization(x_mat[:, 27], 27)  #srv_rerror_rate
MinmaxNormalization(x_mat[:, 28], 28)  #same_srv_rate
MinmaxNormalization(x_mat[:, 29], 29)  #diff_srv_rate
MinmaxNormalization(x_mat[:, 30], 30)  #srv_diff_host_rate

MinmaxNormalization(x_mat[:, 31], 31)  #dst_host_count
MinmaxNormalization(x_mat[:, 32], 32)  #dst_host_srv_count
MinmaxNormalization(x_mat[:, 33], 33)  #dst_host_same_srv_rate
MinmaxNormalization(x_mat[:, 34], 34)  #dst_host_diff_srv_rate
MinmaxNormalization(x_mat[:, 35], 35)  #dst_host_same_src_port_rate
MinmaxNormalization(x_mat[:, 36], 36)  #dst_host_srv_diff_host_rate
MinmaxNormalization(x_mat[:, 37], 37)  #dst_host_serror_rate
MinmaxNormalization(x_mat[:, 38], 38)  #dst_host_srv_serror_rate
MinmaxNormalization(x_mat[:, 39], 39)  #dst_host_rerror_rate
MinmaxNormalization(x_mat[:, 40], 40)  #dst_host_srv_rerror_rate

# 文件写入操作
csv_writer = csv.writer(data_file)
i = 0
while i<len(x_mat[:, 0]):
    csv_writer.writerow(x_mat[i, :])
    i = i + 1
data_file.close()
