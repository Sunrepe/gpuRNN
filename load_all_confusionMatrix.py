import os
import time
import csv
import pandas as pd
import matplotlib
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


def get_one_hot_pre(n_fea):
    file = './data/res10/test/fea{}_kfold0'.format(n_fea)
    res = Read_data_res(file)
    for i_kfold in range(1, 5):
        file = './data/res10/test/fea{}_kfold{}'.format(n_fea, i_kfold)
        _tmp = Read_data_res(file)
        res = np.concatenate([res, _tmp], 0)
    return res


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
    print('PIC:')
    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        res,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    plt.show()
    tick_marks = np.arange(9)
    # plt.yticks(tick_marks, LABELS)
    # plt.savefig('./loss_dir/Matrix_kfold{}.png'.format(k_fold_num), dpi=600, bbox_inches='tight')


def main3():
    '''
    计算不同类输出结果，并作简要分析。
    具体方法：根据不同类进行结果分析。
    :return:
    '''
    # LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    print("------------------------------")
    label_one_hot = Read_data_res('./data/res10/all/label')
    for i_fea in range(9):
        print(stream[i_fea])
        fea_num = i_fea
        one_hot_predictions = get_one_hot_pre(fea_num)
        # Matrix_to_CSV('./data/res10/all/pre_fea{}'.format(fea_num), one_hot_predictions)

        font = {
            'family': 'Times New Roman',
            'weight': 'bold',
            'size': 18
        }
        matplotlib.rc('font', **font)
        #
        width = 12
        height = 12
        plt.figure(figsize=(width, height))

        predictions = one_hot_predictions.argmax(1)
        result_labels = label_one_hot.argmax(1)

        print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

        print("")
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        # Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea{}.txt'.format(fea_num), data=confusion_matrix)
        print(confusion_matrix)

        # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        # width = 12
        # height = 12
        # plt.figure(figsize=(width, height))
        # plt.imshow(
        #     normalised_confusion_matrix,
        #     interpolation='nearest',
        #     cmap=plt.cm.rainbow
        # )
        # plt.title("Confusion matrix \n(normalised to % of total test data)")
        # plt.colorbar()
        # tick_marks = np.arange(10)
        # plt.yticks(tick_marks, LABELS)
        # plt.savefig('./matrix/all1_feas/matrix_fea{}.png'.format(fea_num), dpi=600, bbox_inches='tight')
        print()
        print("------------------------------")
        print()

def main4():
    file = './data/res10/testLabel_kfold0'
    res = Read_data_res(file)
    for i_kfold in range(1, 5):
        file = './data/res10/testLabel_kfold{}'.format(i_kfold)
        _tmp = Read_data_res(file)
        res = np.concatenate([res, _tmp], 0)
    Matrix_to_CSV('./data/res10/all/label', res)


if __name__ == '__main__':
    main3()