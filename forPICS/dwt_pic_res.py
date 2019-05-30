'''
获得相关的作图
main:
    dwt 图像
main2：
    原始信号图像
main3:
    不同人的结果展示
'''

import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
import pywt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

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
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='float', delimiter=",")
    return my_matrix


def get_lei(sq):
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    return alllei.index(sq)


def get_8class(sq):
    # 选择较好的8个动作
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'snap', 'no', 'finger']

    # 选择原先的8个动作
    # alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'snap', 'finger']

    return alllei.index(sq)


def get_label(ch, num_classes=10):
    return np.eye(num_classes)[ch]


# 数据标准化方案1
def z_score(data_x):
    x_m = np.mean(data_x)
    x_p = np.std(data_x)
    x = (data_x-x_m)/x_p
    return x


def getPersons(foldname, kfold_num):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    _person = list(_person)
    _person.sort()
    # _person.remove('marui')
    # _person.remove('zhangyixuan')
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    train_p = []
    for i in _person:
        if i not in test_p:
            train_p.append(i)
    return train_p, test_p


def getPersons_svm(foldname, kfold_num):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob = filename.split('_')
        _person.add(oa)
    _person = list(_person)
    _person.sort()
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    train_p = []
    for i in _person:
        if i not in test_p:
            train_p.append(i)
    return train_p, test_p


def wavelet_trans(data):
    # data = data.T
    wave_let = pywt.Wavelet('sym4')
    data_new = []
    for i in range(8):
        channel_data = data[:, i]

        # 小波变换
        coeffs = pywt.wavedec(channel_data, wavelet=wave_let, level=3)
        new_coeffs = []
        new_coeffs.append(coeffs[0])
        new_coeffs.append(coeffs[1])
        # 只处理高频部分信号.低频信号保留
        for i in range(2, 4):
            i_coeffs = coeffs[i]
            thresh = np.sort(i_coeffs)[int((len(i_coeffs))/2)]/0.6745
            i_coeffs = pywt.threshold(i_coeffs, thresh*3, 'soft', 0)
            new_coeffs.append(i_coeffs)

        # 小波重构
        data_new.append(np.array(pywt.waverec(new_coeffs, wave_let,), 'int'))

    data_new = np.array(data_new)
    return data_new.T


def time_trans(data):
    '''
    主要是对时域做处理,顺便添加最后一项对频域fft进行处理
    :param data:
    :return:
    '''
    res_da = []
    res_da.append(data)
    # mean
    tmp_data = np.zeros(data.shape)
    for i in range(data.shape[0]-4):
        tmp_data[i, :] = np.mean(data[i:i+4, :], axis=0)
    res_da.append(tmp_data[:-4, :])
    # std
    tmp_data = np.zeros(data.shape)
    for i in range(data.shape[0] - 4):
        tmp_data[i, :] = np.std(data[i:i + 4, :], axis=0)
    res_da.append(tmp_data[:-4, :])
    # wave change
    tmp_data = np.zeros(data.shape)
    for i in range(data.shape[0]-1):
        tmp_data[i, :] = data[i+1, :]-data[i, :]
    res_da.append(tmp_data[:-2, :])
    # fft
    tmp_data = np.abs(np.fft.fft(data))
    res_da.append(tmp_data[0:(int(tmp_data.shape[0]/2)+1), :])

    return res_da


def wavelet_trans_data(data):
    wave_let = pywt.Wavelet('sym4')
    t = pywt.wavedec(data, wave_let, level=3, axis=0)
    return t


class AllData_RNN(object):
    '''
    获得训练集与测试集数据,
    主要判断依据是trainable 与 kfold_num
    '''
    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        train_person,test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        if not trainable:print(__person)
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < num_class and oa in __person:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, len(cutting)):
                    if cut == 0:
                        tmp_data = data[0:cutting[cut], :]
                    else:
                        tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                    # _per = [i for i in range(0, tmp_data.shape[0], 4)]
                    # tmp_data = tmp_data[_per, :]
                    tmp_data = tmp_data
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(_len)
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len, :] = tmp_data
                        self.all_data.append(s_tmp)

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')
        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
            self.all_data = self.all_data[_per, :, :]
            self.all_label = self.all_label[_per, :]
            self.all_seq_len = self.all_seq_len[_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        self.all_data = self.all_data[_per, :, :]
        self.all_label = self.all_label[_per, :]
        self.all_seq_len = self.all_seq_len[_per]

    def next(self, batch_size, shuffle=True):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len

def main():
    fold = '../data/tmpdata/'
    data = AllData_RNN(fold)
    da, lab, lens = data.next(1)

    da = da[0][0:int(lens[0]),:]
    wt_da = wavelet_trans_data(da)
    plt.figure()
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        # plt.ylabel('dwt{}'.format(i+1))
        plt.plot(wt_da[i][:, 0:4])
    plt.show()


def main2():
    fold = '../data/tmpdata/'
    data = AllData_RNN(fold)
    da, lab, lens = data.next(1)
    da = da[0][0:int(lens[0]), :]
    plt.figure()
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        # plt.ylabel('dwt{}'.format(i+1))
        plt.plot(da[:, 2*i:(2*i+2)],)
    plt.show()


def main3():
    data = Read_data_res('../all3_everyone_data.txt')
    plt.figure()
    t = [3, 4, 6, 9]
    color_s = [':','--','-.','r-d']

    font1 = {
             'size': 13,
             }
    for i in range(4):
        plt.plot(data[:, t[i]], color_s[i])
    plt.legend(['dwt4', u'原始信号', 'dev', u'组合结果'], prop=font1, loc=4)
    label = []
    tick_marks = np.arange(35)
    for i in range(35):
        label.append(str(i+1))
    # print(label)
    plt.xticks(tick_marks, label)
    # plt.xticks(label)
    plt.title(u'不同被试多流数据识别结果', fontsize=14, fontweight='bold')
    plt.show()

    # plt.savefig('../matrix/all3_everyone.png', dpi=600, bbox_inches='tight')
    pass


if __name__ == '__main__':
    main3()