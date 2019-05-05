import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def f1scores(maxix,rangese):
    recdi = np.sum(maxix,axis=1)
    predi = np.sum(maxix,axis=0)
    print(recdi)
    # print(predi)
    rec = []
    pre = []
    f1sc = []
    for i in range(rangese):
        tmp1 = maxix[i,i]/recdi[i]
        tmp2 = maxix[i,i]/predi[i]
        f1sc.append(2.0/(1.0/tmp1+1.0/tmp2)*10000)
        rec.append(tmp1*10000)
        pre.append(tmp2*10000)
    accu = sum(rec)/rangese
    return [['Recall:',np.array(rec,dtype='int')],['Precision:',np.array(pre,dtype='int')],['F1score:',np.array(f1sc,dtype='int')],accu]



def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def Read_data_res(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 组合的数据
    '''
    my_matrix = np.loadtxt(filename, dtype='float', delimiter=",")
    return my_matrix


def main():
    fold = './matrix/all3kfold0/'
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    matrix_save_path = fold+'Matrix.txt'
    allfile = os.listdir(fold)
    z = np.zeros([100, 10])
    for file in allfile:
        tmp = Read__mean_2(fold+file)
        z += tmp
    z = z.astype('int')
    Matrix_to_CSV(matrix_save_path,z)
    print(z)
    print("------------------------------")
    print("所有的不同特征拓展方案的混淆矩阵")
    # for i in range(10):
    #     print(stream[i])
    #     co = z[i * 10:(i + 1) * 10, :]
    #     print(co)
    for i in range(10):
        print(stream[i])
        co = z[i*10:(i+1)*10, :]
        print(co)
        f1sc = f1scores(co, 10)
        # print(f1sc[0])
        # print(f1sc[1])
        # print(f1sc[2])
        print('Accuracy:{}%'.format(f1sc[3]/100))

        print("------------------------------")
        # Matrix_to_CSV('./data/resd/'+kk+'.txt', co)
        # Matrix_to_CSV('./data/resd/shishi.txt', co)

    # print(type(z))
    # print(type(z[0, 0]))
    # print(z)


def main2():
    '''
    计算所有的cross_errors，
    5 折合并。
    :return:
    '''
    res = np.zeros([9, 9])
    num_class = 10
    for i_kfold in range(5):
        file = './data/cs_errors/2res{}_kfold{}.csv'.format(num_class, i_kfold)
        res += Read_data_res(file)
    Matrix_to_CSV('./data/cs_errors/2res{}_final.csv'.format(num_class), res)


if __name__ == '__main__':
    main2()