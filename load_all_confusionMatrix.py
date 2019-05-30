'''
本函数计算各种matrix，具体每个main函数使用如下：
main1
main2
main3：all_lstm1 最完整结果展示，各类比较及最后的组合结果。
main4
main5
main6
main7
main8：all_lstm1 的pre结果投票成果
main9：all_lstm1 的pre结果叠加成果
main10: all_lstm3的every结果展示，以及最后所有人结果展示
main11：all_lstm3的不同feature结果展示
main12:
    实时识别结果记录。
'''

import os
import time
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

n_classes = 10
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']

def ShowHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('MIC for different Features', y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()


def get_label(ch, num_classes=10):
    return np.eye(num_classes)[ch]


def getPersons_every(foldname):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob = filename.split('_')
        _person.add(ob)
    _person = list(_person)
    _person.sort()
    return _person


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def Matrix_to_CSV_array(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def Matrix_to_CSV_array_append(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def f1scores(maxix, rangese=10):
    recdi = np.sum(maxix, axis=1)
    predi = np.sum(maxix, axis=0)
    # print(recdi)
    # print(predi)
    rec = []
    pre = []
    f1sc = []
    for i in range(rangese):
        tmp1 = maxix[i, i]/recdi[i]
        tmp2 = maxix[i, i]/predi[i]
        f1sc.append(2.0/(1.0/tmp1+1.0/tmp2)*100)
        rec.append(tmp1*100)
        pre.append(tmp2*100)
    accu = sum(rec)/rangese
    return [rec, pre, f1sc], accu
    # return [['Recall:',np.array(rec,dtype='int')],['Precision:',np.array(pre,dtype='int')],['F1score:',np.array(f1sc,dtype='int')],accu]


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


def get_pre_map(pres):
    t_ts = np.zeros([8053, 10])
    for i in range(len(pres)):
        t_ts[i, :] = get_label(pres[i])
    return t_ts


def get_one_hot_pre_kfold(n_fea, n_kfold):
    file = './data/res10/test/fea{}_kfold{}'.format(n_fea, n_kfold)
    res = Read_data_res(file)
    return res


def main():
    '''
    获得all_lstm_3 所有结果拼接的最终result，并作成相关的混淆矩阵
    :return:
    '''
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
    根据5折混淆的最后结果，合并所有的cross_errors
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
    pre_ = []
    rec_ = []
    f1s_ = []
    label_one_hot = Read_data_res('./data/res10/all/label')
    for i_fea in range(1):
        print(stream[i_fea])
        # fea_num = i_fea
        fea_num = 9
        one_hot_predictions = get_one_hot_pre(fea_num)
        Matrix_to_CSV('./data/res10/all/pre_fea{}'.format(fea_num), one_hot_predictions)

        predictions = one_hot_predictions.argmax(1)
        result_labels = label_one_hot.argmax(1)

        pre_.append(metrics.precision_score(result_labels, predictions, average="weighted"))
        rec_.append(metrics.recall_score(result_labels, predictions, average="weighted"))
        f1s_.append(metrics.f1_score(result_labels, predictions, average="weighted"))

        print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

        print("")
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea{}.csv'.format(fea_num), data=confusion_matrix)
        print(confusion_matrix)

        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        plt.imshow(
            normalised_confusion_matrix,
            interpolation='nearest',
            cmap=plt.cm.Blues
        )
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.yticks(tick_marks, LABELS, fontsize=14)
        plt.xticks(tick_marks, LABELS, fontsize=14)
        # plt.show()
        plt.savefig('./matrix/all1_feas/matrix_fea{}.png'.format(fea_num), dpi=300, bbox_inches='tight')
        # print()
        # print("------------------------------")
        # print()
    # df = pd.DataFrame(pre_, index=stream, columns=["Precision"])
    # df["Recall"] = rec_
    # df["F1score"] = f1s_
    # df.to_csv('./matrix/all_lstm/all_tesult.csv')


def main4():
    file = './data/res10/testLabel_kfold0'
    res = Read_data_res(file)
    for i_kfold in range(1, 5):
        file = './data/res10/testLabel_kfold{}'.format(i_kfold)
        _tmp = Read_data_res(file)
        res = np.concatenate([res, _tmp], 0)
    Matrix_to_CSV('./data/res10/all/label', res)


def main5():
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
        for i_kfold in range(5):
            print(stream[i_fea], "Kfold{}".format(i_kfold))
            one_hot_pre = get_one_hot_pre_kfold(fea_num, i_kfold)
            label = Read_data_res('./data/res10/testLabel_kfold{}'.format(i_kfold))
            predictions = one_hot_pre.argmax(1)
            result_labels = label.argmax(1)

            print(
                "Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
            print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
            print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

            if metrics.recall_score(result_labels, predictions, average="weighted") < 0.2:
                print("Here Wrong ----------------------------------")
            print("Confusion Matrix:")
            confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
            # Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea{}.txt'.format(fea_num), data=confusion_matrix)
            print(confusion_matrix)

        print("\n----Final Result----{}".format(stream[i_fea]))
        one_hot_predictions = get_one_hot_pre(fea_num)
        # Matrix_to_CSV('./data/res10/all/pre_fea{}'.format(fea_num), one_hot_predictions)

        # font = {
        #     'family': 'Times New Roman',
        #     'weight': 'bold',
        #     'size': 18
        # }
        # matplotlib.rc('font', **font)
        # #
        # width = 12
        # height = 12
        # plt.figure(figsize=(width, height))

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


def main6():
    '''
    计算所有的cross_errors，
    该处结果是MIC结果
    5 折合并。
    :return:
    '''
    res = np.zeros([9, 9])
    num_class = 10
    for i_kfold in range(5):
        file = './data/cs_errors/4res{}_kfold{}.csv'.format(num_class, i_kfold)
        res += Read_data_res(file)
    Matrix_to_CSV('./data/cs_errors/4res{}_final.csv'.format(num_class), res)
    print('PICS')
    # Plot Results:
    ShowHeatMap(res/5.0)


def main7():
    '''
    计算最佳结果：
        即应用所有特征进行分类，只要有一类正确便视为正确分类，看最后结果。
    vote for result!
    :return:
    '''
    # LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    print("------------------------------")
    label_one_hot = Read_data_res('./data/res10/all/label')
    t_trues = np.zeros(8053)
    youxiao_neirong = [0,1,2,3,4,5,6,7]
    for i_fea_i in range(8):
        i_fea = youxiao_neirong[i_fea_i]
        # i_fea = i_fea_i
        print(stream[i_fea])
        fea_num = i_fea
        one_hot_predictions = get_one_hot_pre(fea_num)

        predictions = one_hot_predictions.argmax(1)
        result_labels = label_one_hot.argmax(1)
        x_tmp = predictions==result_labels
        t_trues += x_tmp
    one_hot_predictions = get_one_hot_pre(0).argmax(1)
    s_2 = np.zeros(8053, dtype=np.int)
    s_1 = label_one_hot.argmax(1)
    for i in range(8053):
        if t_trues[i] > 1:
            # print('S2:{}, type:{}'.format(s_2[i], type(s_2[i])))
            # print('S1:{}, type:{}'.format(s_1[i], type(s_1[i])))
            s_2[i] = s_1[i]
        else:
            s_2[i] = one_hot_predictions[i]
            # # print('wrong!')
            # if s_1[i] == 9:
            #     # s_2[i] = s_1[i]-1
            #     pass
            # else:
            #     # s_2[i] = s_1[i]+1
            #     s_2[i] = one_hot_predictions[i]
    print("Precision: {}%".format(100 * metrics.precision_score(s_1, s_2, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(s_1, s_2, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(s_1, s_2, average="weighted")))

    print("")
    print("Confusion Matrix7:")
    confusion_matrix = metrics.confusion_matrix(s_1, s_2,)
    Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea-zuijia.txt', data=confusion_matrix)
    print(confusion_matrix)


def main8():
    '''
    计算vote 结果:
        即应用所有特征进行分类，只要有一类正确便视为正确分类，看最后结果。
    vote for result!
    :return:
    '''
    print("all_lstm1投票结果")
    # LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    print("------------------------------")
    label_one_hot = Read_data_res('./data/res10/all/label')
    t_trues = np.zeros([8053, 10])

    for i_fea in range(9):
        print(stream[i_fea])
        fea_num = i_fea
        one_hot_predictions = get_one_hot_pre(fea_num)
        predictions = one_hot_predictions.argmax(1)
        t_trues += get_pre_map(predictions)

    s_2 = t_trues.argmax(1)
    s_1 = label_one_hot.argmax(1)

    print("Precision: {}%".format(100 * metrics.precision_score(s_1, s_2, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(s_1, s_2, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(s_1, s_2, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(s_1, s_2,)
    Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea-zuijia.txt', data=confusion_matrix)
    print(confusion_matrix)


def main9():
    '''
    计算最后结果--叠加结果：
        即应用所有特征进行分类，只要有一类正确便视为正确分类，看最后结果。
    vote for result!
    :return:
    '''
    print("输出叠加结果：")
    # LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    print("------------------------------")
    label_one_hot = Read_data_res('./data/res10/all/label')
    t_trues = np.zeros([8053, 10])

    for i_fea in range(9):
        print(stream[i_fea])
        fea_num = i_fea
        one_hot_predictions = get_one_hot_pre(fea_num)
        t_trues += one_hot_predictions

    s_2 = t_trues.argmax(1)
    s_1 = label_one_hot.argmax(1)

    print("Precision: {}%".format(100 * metrics.precision_score(s_1, s_2, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(s_1, s_2, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(s_1, s_2, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(s_1, s_2,)
    Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea-zuijia.txt', data=confusion_matrix)
    print(confusion_matrix)


def main10():
    '''
    计算不同类输出结果，并作简要分析。
        根据不同人做结果输出。
        具体方法：根据不同类进行结果分析。
    :return:
    '''
    fold = './data/res10/all_lstm3/'

    pre_ = []
    rec_ = []
    f1s_ = []
    all_pre = []
    all_label = []
    for person in getPersons_every(fold):
        print("------------------------------")
        print(person)
        label_one_hot = Read_data_res(fold+'label_{}'.format(person))

        one_hot_predictions = Read_data_res(fold+'pre_{}'.format(person))

        predictions = one_hot_predictions.argmax(1)
        result_labels = label_one_hot.argmax(1)
        for pre_i in predictions: all_pre.append(pre_i)
        for label_i in result_labels: all_label.append(label_i)

        pre_.append(metrics.precision_score(result_labels, predictions, average="weighted"))
        rec_.append(metrics.recall_score(result_labels, predictions, average="weighted"))
        f1s_.append(metrics.f1_score(result_labels, predictions, average="weighted"))

        print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

        print("")
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        # Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea{}.txt'.format(fea_num), data=confusion_matrix)
        print(confusion_matrix)
        #
        # print()
        # print("------------------------------")
        print()
    print("------------------------------")
    print('最终结果：')
    pre_.append(metrics.precision_score(all_label, all_pre, average="weighted"))
    rec_.append(metrics.recall_score(all_label, all_pre, average="weighted"))
    f1s_.append(metrics.f1_score(all_label, all_pre, average="weighted"))

    print("Precision: {}%".format(100 * metrics.precision_score(all_label, all_pre, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(all_label, all_pre, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(all_label, all_pre, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(all_label, all_pre,)
    Matrix_to_CSV_array('./matrix/all_3/matrix_final.csv', confusion_matrix)
    print(confusion_matrix)

    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.yticks(tick_marks, LABELS)
    plt.xticks(tick_marks, LABELS)
    plt.show()
    # plt.savefig('./matrix/all_3/matrix_all.png', dpi=600, bbox_inches='tight')

    # stream = getPersons_every(fold)
    # stream.append('最终结果')
    # df = pd.DataFrame(pre_, index=stream, columns=["Precision"])
    # df["Recall"] = rec_
    # df["F1score"] = f1s_
    # df.to_csv('./matrix/all_3/all_tesult.csv')


def main11():
    '''
    计算不同类输出结果，并作简要分析。
        具体方法：根据不同特征拓展方案进行结果分析。
    :return:
    '''
    print("All lstm3 结果展示：")
    fold = './data/res10/all_lstm3/'
    filelist = getPersons_every(fold)
    pre_ = []
    rec_ = []
    f1s_ = []
    fold = './data/res10/all_lstm3_feas/'

    for i_fea in [9]:
    # for i_fea in range(10):
        print("------------------------------")
        print('All_fea{}'.format(i_fea))
        all_pre = []
        all_label = []
        for person in filelist:
            label_one_hot = Read_data_res(fold+'label_{}'.format(person))

            one_hot_predictions = Read_data_res(fold+'pre_fea{}_{}'.format(i_fea, person))

            predictions = one_hot_predictions.argmax(1)
            result_labels = label_one_hot.argmax(1)
            for pre_i in predictions: all_pre.append(pre_i)
            for label_i in result_labels: all_label.append(label_i)

        pre_.append(metrics.precision_score(all_label, all_pre, average="weighted"))
        rec_.append(metrics.recall_score(all_label, all_pre, average="weighted"))
        f1s_.append(metrics.f1_score(all_label, all_pre, average="weighted"))

        print("Precision: {}%".format(100 * metrics.precision_score(all_label, all_pre, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(all_label, all_pre, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(all_label, all_pre, average="weighted")))

        print("")
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(all_label, all_pre)

        Matrix_to_CSV_array('./matrix/all_3/matrix_final-feas9.csv', confusion_matrix)
        print(confusion_matrix)
        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        plt.imshow(
            normalised_confusion_matrix,
            interpolation='nearest',
            cmap=plt.cm.Blues
        )
        # plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.yticks(tick_marks, LABELS, fontsize=14)
        plt.xticks(tick_marks, LABELS, fontsize=14)
        # plt.show()
        plt.savefig('./matrix/all_3/matrix_final-feas9.png', dpi=300, bbox_inches='tight')

    # ------------------ feas ------------------
    # stream = []
    # for i in range(10):
    #     stream.append('fea{}'.format(i))
    # df = pd.DataFrame(pre_, index=stream, columns=["Precision"])
    # df["Recall"] = rec_
    # df["F1score"] = f1s_
    # df.to_csv('./matrix/all_3/all3_feas_tesult.csv')
    # ------------------ feas ------------------


def main12():
    file = 'realtime_act-resandpre.txt'
    data = Read__mean_2(file)
    predictions = data[:, 1]
    result_labels = data[:, 0]

    print("len_pre:{},\tlen_label:{}".format(len(predictions), len(result_labels)))
    print("最终结果：")
    print(
        "Precision: {}%".format(
            100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    print("")
    print("Confusion Matrix:")
    print(confusion_matrix)

    Matrix_to_CSV_array_append('./matrix/realtime_matrix.csv', confusion_matrix)
    sc, acc = f1scores(confusion_matrix, 10)
    Matrix_to_CSV_array_append('./matrix/realtime_matrix.csv', np.array(sc))

    print("Accuracy: {}".format(acc))
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    # ShowHeatMap(normalised_confusion_matrix)
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.yticks(tick_marks, LABELS)
    plt.xticks(tick_marks, LABELS)
    plt.show()


if __name__ == '__main__':
    main3()
