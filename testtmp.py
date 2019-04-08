'''
采用小波变换去噪,
先变换,去噪后重构原始信号.
'''

import os
import csv
import time
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pywt


def Matrix_to_CSV(filename, data):
    import numpy as np
    data = data.astype('int')
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def wave_change(data):
    _len = data.shape[0]
    for i in range(_len - 1):
        data[i, :] = data[i + 1, :] - data[i, :]
    return data[0:-1, :]


def partition(num, low, high):
    pivot = num[low]
    while(low<high):
        while(low < high and num[high] > pivot):
            high -= 1
        while (low < high and num[low] < pivot):
            low += 1
        temp = num[low]
        num[low] = num[high]
        num[high] = temp
    num[low] = pivot
    return low


def findkth(num, low, high, k):  # 找到数组里第k个数
    index = partition(num, low, high)
    if index == k: return num[index]
    if index < k:
        return findkth(num, index + 1, high, k)
    else:
        return findkth(num, low, index - 1, k)

# data_n = wavelet_trans(data)

def wavelet_trans(data):
    # data = data.T
    wave_let = pywt.Wavelet('db2')
    data_new = []
    for i in range(8):
        channel_data = data[:,i]
        # 小波变换
        coeffs = pywt.wavedec(channel_data, wavelet=wave_let, level=3)
        new_coeffs = []
        for i_coeffs in coeffs:
            # print(findkth(pai, 0, len(pai) - 1, 0))
            thresh = np.sort(i_coeffs)[int((len(i_coeffs))/2)]/0.6745
            i_coeffs = pywt.threshold(i_coeffs,thresh,'soft',0)
            new_coeffs.append(i_coeffs)
        # 小波重构
        data_new.append(np.array(pywt.waverec(new_coeffs, wave_let,),'int'))
    data_new = np.array(data_new)
    return data_new.T

            # print(pywt.dwt_max_level(50, wave_let))      # 查看可进行的最高分解层次
        # print(np.array(pywt.waverec(coeffs, 'db5'), 'int'))  # 查看反小波分解
        # print('coffs.shaoe',coeffs[0])

def main():
    # foldname = '../myodata/actdata/'
    foldname = '../myodata/testdata/'
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        if oc == 'b.txt':
            print(filename)
            a_filename = '../../gpuRNN/data/wtdata/' + oa + '_' + ob + '_c.txt'
            csvfile = open(a_filename, "a", newline='')
            writer = csv.writer(csvfile)

            filenames = foldname + filename
            data = Read__mean_2(filenames)
            cutting = np.loadtxt(foldname + oa + '_' + ob + '_c.txt')
            _last = 0
            for i in range(0, len(cutting)):
                tmp_data = data[_last:int(_last + cutting[i]), :]
                _last = int(_last + cutting[i])
                tmp_data = wavelet_trans(tmp_data)
                Matrix_to_CSV('../../gpuRNN/data/wtdata/' + oa + '_' + ob + '_b.txt', tmp_data)
                writer.writerow([_last])


if __name__ == '__main__':
    main()

