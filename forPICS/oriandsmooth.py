import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签


def smooth_data_middle(data_smo, pin_long=5, pin_times = 3):
    '''
    对于一维数据进行数据平滑
    :param data_smo: 原始一维数据
    :param pin_long: 平滑窗长度
    :param pin_times: 平滑次数
    :return:
    '''
    all_long = len(data_smo)
    # print('长', all_long, 'times', pin_times)
    # print(data_smo.shape)
    if all_long < pin_long:
        # print("The data you input is too short to smooth!")
        exit()
        return 0
    else:
        qian_mian_chang = int(pin_long/2)
        hou_mian_chang = pin_long - qian_mian_chang - 1
        for i in range(pin_times):
            # print('平滑次数', i+1)
            for index in range(qian_mian_chang):
                data_smo[index] = np.mean(data_smo[0:index+1])
            for index in range(qian_mian_chang, all_long-hou_mian_chang):
                data_smo[index] = np.mean(data_smo[index-qian_mian_chang:index+1+hou_mian_chang])
            for index in range(all_long-hou_mian_chang, all_long-1):
                data_smo[index] = np.mean(data_smo[index:all_long])
        return data_smo


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def Read__data(filename):
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def main():
    file = './tmp_ori.txt'
    data = Read__data(file)

    no_1000 = []
    for i in range(data.shape[0]):
        if data[i, 0] == 1000:
            no_1000.append(i)

    ds = [145, 580, 1020, 1487]
    data[data > 128] = 0
    dt = np.mean(np.abs(data), 1)
    dt = smooth_data_middle(dt, 80, 2)


    plt.figure()
    plt.title(u"预处理及动作分割")
    l1, = plt.plot(data[:, 0], color='silver')
    l2, = plt.plot(dt, color='orange')
    for i in range(3):
        p2, = plt.plot(ds[i], dt[ds[i]], marker='*', mec='r', mfc='w', ms=10)
        p1, = plt.plot(no_1000[i] + 40, dt[no_1000[i] + 40], marker='o', mec='r', mfc='w')
    p2, = plt.plot(ds[3], dt[ds[3]], marker='*', mec='r', mfc='w', ms=10)
    p1, = plt.plot(no_1000[3] + 40, dt[no_1000[3] + 40], marker='o', mec='r', mfc='w')

    plt.legend([l1, l2, p2, p1], [u"原始信号", u"平滑信号", u"动作起点", u"动作终点"])
    plt.show()


if __name__ == '__main__':
    main()