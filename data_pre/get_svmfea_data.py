'''
用于SVM_OLD 方法结果
main：
    用户独立
main2：
    用户依赖_self
main3:
    用户依赖，但是只使用均值。非时间序列！
'''

import numpy as np
import os
import csv


def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def data_len_get(longs, kfold_num):
    l = int(longs*kfold_num/5.0)
    r = int(longs*(kfold_num+1)/5.0)
    return l, r


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def get_lei(sq):
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    return alllei.index(sq)


def getPersons_every(foldname):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    _person = list(_person)
    _person.sort()
    return _person


def z_score(data_x):
    x_m = np.mean(data_x)
    x_p = np.std(data_x)
    x = (data_x-x_m)/x_p
    return x


def get_2emg(data_d):
    data_len = data_d.shape[0]
    data_mid = int(data_len/5)
    x = []
    for i in range(8):
        # ------5-----------
        x.append(np.mean(data_d[0:data_mid*2, i]))
        x.append(np.mean(data_d[data_mid:data_mid*3, i]))
        x.append(np.mean(data_d[data_mid*2:data_mid*4, i]))
        # x.append(np.mean(data_d[data_mid*3:data_mid*5, i]))
        x.append(np.mean(data_d[data_mid*3:data_len, i]))
        # ------5-----------

        # ------4-----------
        # x.append(np.mean(data_d[0:data_mid * 1, i]))
        # x.append(np.mean(data_d[data_mid:data_mid * 2, i]))
        # x.append(np.mean(data_d[data_mid * 2:data_mid * 3, i]))
        # x.append(np.mean(data_d[data_mid * 3:data_len, i]))
        # ------4-----------
    # print(x)
    return np.array(x)


def main():
    fold = '../data/actdata/'
    acts = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    all_person = getPersons_every(fold)
    for person in all_person:
        print("Now {}".format(person))
        data_mean = []
        for act in acts:
            oa, ob = person, act
            filename = fold + oa + '_' + ob + '_b.txt'
            data = Read__mean_2(filename)
            cutting = Read__mean_2(fold + oa + '_' + ob + '_c.txt')
            for cut in range(0, len(cutting)):
                if cut == 0:
                    tmp_data = data[0:cutting[cut], :]
                else:
                    tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                dongzuo_data = z_score(get_2emg(np.abs(tmp_data)))

                if np.isnan(dongzuo_data).any():  # or get_lei(ob)>6  or duan_data.shape[0]<60
                    print('nan:', filename)
                    pass
                else:
                    dongzuo_data = np.append(dongzuo_data, [get_lei(ob)])
                    data_mean.append(dongzuo_data)
        mean_d = np.array(data_mean)
        new_file = person + '_svmfeas.txt'
        Matrix_to_CSV('../data/data_svmfea/' + new_file, mean_d)


def main2():
    fold = '../data/actdata/'
    acts = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    all_person = getPersons_every(fold)
    for i_kfold in range(5):
        data_mean_train = []
        data_mean_test = []
        print("Now kfold{}".format(i_kfold))
        for person in all_person:
            for act in acts:
                oa, ob = person, act
                filename = fold + oa + '_' + ob + '_b.txt'
                data = Read__mean_2(filename)
                cutting = Read__mean_2(fold + oa + '_' + ob + '_c.txt')
                _starts, _ends = data_len_get(len(cutting), i_kfold)  # 确定该区间范围
                for cut in range(0, len(cutting)):
                    if cut >= _starts and cut < _ends:
                        # test--
                        if cut == 0:
                            tmp_data = data[0:cutting[cut], :]
                        else:
                            tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                        dongzuo_data = z_score(get_2emg(np.abs(tmp_data)))
                        dongzuo_data = np.append(dongzuo_data, [get_lei(ob)])
                        data_mean_test.append(dongzuo_data)
                    else:
                        # train--
                        if cut == 0:
                            tmp_data = data[0:cutting[cut], :]
                        else:
                            tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                        dongzuo_data = z_score(get_2emg(np.abs(tmp_data)))
                        dongzuo_data = np.append(dongzuo_data, [get_lei(ob)])
                        data_mean_train.append(dongzuo_data)
        mean_d_train = np.array(data_mean_train)
        mean_d_test = np.array(data_mean_test)
        Matrix_to_CSV('../data/data_svmfea_self/train_fold{}.txt'.format(i_kfold), mean_d_train)
        Matrix_to_CSV('../data/data_svmfea_self/test_fold{}.txt'.format(i_kfold), mean_d_test)


def main3():
    fold = '../data/actdata/'
    acts = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    all_person = getPersons_every(fold)
    for i_kfold in range(5):
        data_mean_train = []
        data_mean_test = []
        print("Now kfold{}".format(i_kfold))
        for person in all_person:
            for act in acts:
                oa, ob = person, act
                filename = fold + oa + '_' + ob + '_b.txt'
                data = Read__mean_2(filename)
                cutting = Read__mean_2(fold + oa + '_' + ob + '_c.txt')
                _starts, _ends = data_len_get(len(cutting), i_kfold)  # 确定该区间范围
                for cut in range(0, len(cutting)):
                    if cut >= _starts and cut < _ends:
                        # test--
                        if cut == 0:
                            tmp_data = data[0:cutting[cut], :]
                        else:
                            tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                        # dongzuo_data = z_score(get_2emg(np.abs(tmp_data)))
                        dongzuo_data = z_score(np.mean(np.abs(tmp_data), axis=0))
                        dongzuo_data = np.append(dongzuo_data, [get_lei(ob)])
                        data_mean_test.append(dongzuo_data)
                    else:
                        # train--
                        if cut == 0:
                            tmp_data = data[0:cutting[cut], :]
                        else:
                            tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                        # dongzuo_data = z_score(get_2emg(np.abs(tmp_data)))
                        dongzuo_data = z_score(np.mean(np.abs(tmp_data), axis=0))
                        dongzuo_data = np.append(dongzuo_data, [get_lei(ob)])
                        data_mean_train.append(dongzuo_data)
        mean_d_train = np.array(data_mean_train)
        mean_d_test = np.array(data_mean_test)
        Matrix_to_CSV('../data/data_svmfea_self/mean_train_fold{}.txt'.format(i_kfold), mean_d_train)
        Matrix_to_CSV('../data/data_svmfea_self/mean_test_fold{}.txt'.format(i_kfold), mean_d_test)


if __name__ == '__main__':
    main3()