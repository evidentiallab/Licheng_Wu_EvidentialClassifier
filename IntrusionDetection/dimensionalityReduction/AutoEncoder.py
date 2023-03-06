import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, Activation
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import time


def train_AE_121_100(data, is_train):
    # 定义自编码器模型
    input_data = Input(shape=(121,))
    hidden_layer = Dense(100, activation='relu')(input_data)
    output_data = Dense(121, activation='sigmoid')(hidden_layer)
    autoencoder = Model(inputs=input_data, outputs=output_data)
    encoder = Model(inputs=input_data, outputs=hidden_layer)
    # autoencoder.summary()
    if is_train == 1:
        # 训练自编码器模型
        autoencoder.compile(optimizer='adam', loss='mse')
        h = autoencoder.fit(data, data, epochs=20, batch_size=128)
        autoencoder.save('AE121-100(MinMax).h5')

        history = h.history
        epochs = range(len(history['loss']))
        plt.plot(epochs, history['loss'])
        plt.xlabel("Epochs")
        plt.ylabel("Reconstruction Error")
        plt.rcParams["figure.dpi"] = 300
        plt.savefig('../pic/AE121-100.png', dpi=300)
        plt.show()

    if is_train == 0:
        # 使用自编码器模型进行预测
        AE = load_model('AE121-100(MinMax).h5')
        enc_data = encoder.predict(data)
        # pd.DataFrame(enc_data).to_csv('../dataset/KDDCUP99/Encoded/121_100.csv')
        dec_data = AE.predict(data)
        pd.DataFrame(dec_data).to_csv('../dataset/KDDCUP99/Encoded/121_100_test.csv')
        # print(enc_data.shape)


if __name__ == '__main__':
    # df = pd.read_csv('../dataset/KDDCUP99/Z-score_121.csv', header=None)
    df = pd.read_csv('../dataset/KDDCUP99/MinMax/feature_121.csv')
    dataND = df.values
    # label = dataND[:, 121:]
    dataNd = dataND[:, 0:121]

    is_train = 1
    train_AE_121_100(dataNd,is_train)
    # check_AE(dataNd[0,:])


