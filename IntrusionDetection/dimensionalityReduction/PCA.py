from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
""" 
Notes:
cumulativeVarRate(81/121) = 0.9427558796328908
cumulativeVarRate(100/121) = 0.9993533658742644
cumulativeVarRate(121/121) = 0.9999999999999998
cumulativeVarRate(81/124) = 0.9359615729174101
cumulativeVarRate(100/124) = 0.9991938700513163
cumulativeVarRate(121/124) = 1.0
"""

def basicPCA(n, data):
    print(type(data))
    pca = PCA(n_components=n)
    newX = pca.fit_transform(data)
    invX = pca.inverse_transform(newX)
    pd.DataFrame(invX).to_csv('../dataset/KDDCUP99/inverseTrans/121_124(dummy).csv')
    # print(type(pca.explained_variance_ratio_))
    s = pca.explained_variance_ratio_
    print('cumulativeVarRate = ' + str(s.sum()))
    return s


def plotCumulativeVarRatio(n, data):
    x = []
    y = []
    rate_list = basicPCA(n, data)
    s = 0
    for i in range(0,n):
        x.append(i)
        y.append(s)
        s += rate_list[i]
        print(y)
        print('cumulating ' + str(i) + ' principal components')
    plt.plot(x, y)
    plt.xlabel("number of components")
    plt.ylabel("cumulative variance contribution rate")
    plt.rcParams["figure.dpi"] = 300
    plt.savefig('../pic/CVCR-Zscore(124dummy).png',dpi=300)
    plt.show()


if __name__ == '__main__':
    n_feature = 124
    pc = 121

    if n_feature == 121:
        df = pd.read_csv('../dataset/KDDCUP99/Z-score_121(dummy).csv', header=None)
        df.fillna(0, inplace=True)
        data = df.iloc[:, 0:121]
        dataNd = data.values

        # print(dataNd)
        # print(type(dataNd))
        # print(dataNd.shape)
        basicPCA(pc, dataNd)
        # plotCumulativeVarRatio(n_feature, dataNd)


    if n_feature == 124:
        df = pd.read_csv('../dataset/KDDCUP99/Z-score_124(dummy).csv', header=None)
        df.fillna(0, inplace=True)
        data = df.iloc[:, 0:124]
        dataNd = data.values
        # print(data)
        # print(np.isnan(data).any().to_string())
        basicPCA(pc, dataNd)
        # plotCumulativeVarRatio(n_feature, dataNd)
