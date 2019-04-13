'''
测试数据融合结果
用EMG信号最后输出
以及wavelet的mean_pool
'''
from data_pre.alldata import *
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tmp_trans_wavelet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warnings and errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = 150  # Loop 1000 times on the dataset
batch_size = 100
display_iter = 1000  # To show test set accuracy during training
model_save = 80

k_fold_num = 0
fold = './data/actdata/'
savename = '_LSTM_emgandwavelet_kfold'+str(k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']

def show(ori_func, ft, sampling_period = 5):
    n = len(ori_func)
    interval = sampling_period / n
    # 绘制原始函数
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, sampling_period, interval), ori_func, 'black')
    plt.xlabel('Time'), plt.ylabel('Amplitude')
    # 绘制变换后的函数
    plt.subplot(2,1,2)
    frequency = np.arange(n / 2) / (n * interval)
    nfft = abs(ft[range(int(n / 2))] / n )
    plt.plot(frequency, nfft, 'red')
    plt.xlabel('Freq (Hz)'), plt.ylabel('Amp. Spectrum')
    plt.show()


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = waveandemg_RNNData(foldname=fold, max_seq=max_seq,
                             num_class=n_classes, trainable=True, kfold_num=k_fold_num)
    data = train_sets.all_data[0]
    _len = int(train_sets.all_seq_len[0])
    x = data[0:_len, 0]
    y = np.fft.fft(x)
    show(x, y)
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(x)
    # plt.subplot(212)
    # plt.plot(y)
    # plt.show()

if __name__ == '__main__':
    main()