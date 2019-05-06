import numpy as np
import os
import pywt

def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
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


def data_len_get(longs, kfold_num):
    l = int(longs*kfold_num/5.0)
    r = int(longs*(kfold_num+1)/5.0)
    return l, r


def get_last_dconvlen(real_len):
    return int(int((int((real_len-5)/2)-9)/2-9)/2)-4


def fft_trans(data):
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

def data_feature_transform(data, feature):
    '''
    对所有特征提取，并根据feature类型选择返回，其中-1表示全部返回
    :param data:
    :param feature:
    :return:
    '''
    res_da = []
    if feature == 0:
        res_da.append(data)  # 0
        return res_da
    # mean 1
    elif feature == 1:
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 4):
            tmp_data[i, :] = np.mean(data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-4, :])
        return res_da
    # std 2
    elif feature == 2:
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 4):
            tmp_data[i, :] = np.std(data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-4, :])
        return res_da
    # wave change 3
    elif feature == 3:
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 1): # calculate
            tmp_data[i, :] = data[i + 1, :] - data[i, :]
        for i in range(data.shape[0] - 5):  # calculate
            tmp_data[i, :] = np.std(tmp_data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-5, :])
        return res_da
    # 频域
    elif feature == 4:
        wave_let = pywt.Wavelet('sym4')
        t = pywt.wavedec(data, wave_let, level=3, axis=0)
        res_da.append(t[0])
        return res_da
    elif feature == 5:
        wave_let = pywt.Wavelet('sym4')
        t = pywt.wavedec(data, wave_let, level=3, axis=0)
        res_da.append(t[0])
        return res_da
    elif feature == 6:
        wave_let = pywt.Wavelet('sym4')
        t = pywt.wavedec(data, wave_let, level=3, axis=0)
        res_da.append(t[0])
        return res_da
    elif feature == 7:
        wave_let = pywt.Wavelet('sym4')
        t = pywt.wavedec(data, wave_let, level=3, axis=0)
        res_da.append(t[0])
        return res_da
    # fft  8
    elif feature == 8:
        tmp_data = np.abs(np.fft.fft(data))
        res_da.append(tmp_data[0:(int(tmp_data.shape[0] / 2) + 1), :])
        return res_da

    # all in
    if feature == -1:
        res_da.append(data)  # 0
        # mean 1
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 4):
            tmp_data[i, :] = np.mean(data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-4, :])
        # std 2
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 4):
            tmp_data[i, :] = np.std(data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-4, :])

        # wave change 3
        tmp_data = np.zeros(data.shape)
        for i in range(data.shape[0] - 1):  # calculate
            tmp_data[i, :] = data[i + 1, :] - data[i, :]
        for i in range(data.shape[0] - 5):  # calculate
            tmp_data[i, :] = np.std(tmp_data[i:i + 4, :], axis=0)
        res_da.append(tmp_data[:-5, :])

        # 频域
        wave_let = pywt.Wavelet('sym4')
        t = pywt.wavedec(data, wave_let, level=3, axis=0)
        dwt1, dwt2, dwt3, dwt4 = t
        res_da.append(dwt1)  # 4
        res_da.append(dwt2)  # 5
        res_da.append(dwt3)  # 6
        res_da.append(dwt4)  # 7

        # fft  8
        tmp_data = np.abs(np.fft.fft(data))
        res_da.append(tmp_data[0:(int(tmp_data.shape[0] / 2) + 1), :])
        return res_da

# class All_data_merge()
class All_data_feature_tmp(object):
    '''
    获得9种特征用于最后测试
    '''
    def __init__(self, foldname, max_seq=800, num_class=10, trainable=False, kfold_num=0, feature_num=0):
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
        # __person = ['zhouxufeng']

        print(__person)
        if not trainable:
            print(__person)
        self.batch_id = 0  # use for batch_get
        self.all_label = []  # only one use
        # 对应 dwt4层,ori,mean,std,zeros
        tmp_use_len = [800, 800, 800, 800, 150, 150, 250, 450, 400]
        self.data = {}
        self.seqlen = {}
        # 开始生产序列信号,存储于字典中
        for i_initdict in range(len(tmp_use_len)):
            self.data[i_initdict] = []
            self.seqlen[i_initdict] = []
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
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        # others data
                        feature_data = data_feature_transform(tmp_data, feature_num)
                        print('len fea:', len(feature_data))
                        for i_coeffs in range(len(feature_data)):
                            _len = feature_data[i_coeffs].shape[0]
                            self.seqlen[i_coeffs].append(_len)
                            s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
                            s_tmp[0:_len, :] = feature_data[i_coeffs]
                            self.data[i_coeffs].append(s_tmp)


        for i_toarray in range(len(tmp_use_len)):
            self.seqlen[i_toarray] = np.array(self.seqlen[i_toarray]).astype('float32')
            self.data[i_toarray] = np.array(self.data[i_toarray]).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')

        print(self.all_label.shape)
        print(self.data[0].shape)

        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
            self.all_label = self.all_label[_per, :]
            for i_shuffle in range(len(tmp_use_len)):
                self.data[i_shuffle] = self.data[i_shuffle][_per, :, :]
                self.seqlen[i_shuffle] = self.seqlen[i_shuffle][_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
        self.all_label = self.all_label[_per, :]
        for i_shuffle in range(9):
            self.data[i_shuffle] = self.data[i_shuffle][_per, :, :]
            self.seqlen[i_shuffle] = self.seqlen[i_shuffle][_per]

    def next(self, batch_size, shuffle=True):
        if self.batch_id == len(self.all_label):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_label))]
        batch_data = []
        batch_seq_len = []
        for i_batch in range(9):
            batch_seq_len.append(self.seqlen[i_batch][self.batch_id:min(self.batch_id + batch_size, len(self.all_label))])
            batch_data.append(self.data[i_batch][self.batch_id:min(self.batch_id + batch_size, len(self.all_label))])
        self.batch_id = min(self.batch_id + batch_size, len(self.all_label))
        return batch_labels, batch_data, batch_seq_len


class All_data_feature(object):
    '''
    获得训练集与测试集数据,
    主要判断依据是trainable 与 kfold_num
    '''
    def __init__(self, foldname, max_seq=700, num_class=10,
                 trainable=False, kfold_num=0, feature_num=0):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        # print(__person)
        print("feature NUM:", feature_num )
        if not trainable:
            print('testPersons:', __person)
        else:
            print('trainPersons:', __person)
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
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        tmp_data = data_feature_transform(tmp_data, feature_num)[0]
                        _len = tmp_data.shape[0]
                        tmp_data = z_score(tmp_data)
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


class All_data_feature_test(object):
    '''
    获得训练集与测试集数据,
    主要判断依据是trainable 与 kfold_num
    '''
    def __init__(self, foldname, max_seq=800, num_class=10,
                 trainable=False, kfold_num=0, feature_num=0):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        # print(__person)
        # __person = ['zhouxufeng']
        print("feature NUM:", feature_num)
        if not trainable:
            print('testPersons:', __person)
        else:
            print('trainPersons:', __person)
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
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        tmp_data = data_feature_transform(tmp_data, feature_num)[0]
                        _len = tmp_data.shape[0]
                        tmp_data = z_score(tmp_data)
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(_len)
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len, :] = tmp_data
                        self.all_data.append(s_tmp)

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')

        # 打乱数据
        # if trainable:
        #     _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        #     self.all_data = self.all_data[_per, :, :]
        #     self.all_label = self.all_label[_per, :]
        #     self.all_seq_len = self.all_seq_len[_per]

    # def _shuffle_data(self):
    #     _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
    #     self.all_data = self.all_data[_per, :, :]
    #     self.all_label = self.all_label[_per, :]
    #     self.all_seq_len = self.all_seq_len[_per]

    def next(self, batch_size, shuffle=True):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            # if shuffle:
            #     self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len


class All_data_feature_lable(object):
    '''
    获得训练集与测试集数据,
    主要判断依据是trainable 与 kfold_num
    '''
    def __init__(self, foldname, max_seq=800, num_class=10,
                 trainable=False, kfold_num=0, feature_num=0):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        # print(__person)
        # __person = ['zhouxufeng']
        print("feature NUM:", feature_num)
        if not trainable:
            print('testPersons:', __person)
        else:
            print('trainPersons:', __person)
        self.all_label = []
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
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        # tmp_data = data_feature_transform(tmp_data, feature_num)[0]
                        # _len = tmp_data.shape[0]
                        # tmp_data = z_score(tmp_data)
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        # self.all_seq_len.append(_len)
                        # s_tmp = np.zeros((max_seq, 8))
                        # s_tmp[0:_len, :] = tmp_data
                        # self.all_data.append(s_tmp)

        # self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        # self.all_seq_len = np.array(self.all_seq_len).astype('float32')

        # 打乱数据
        # if trainable:
        #     _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        #     self.all_data = self.all_data[_per, :, :]
        #     self.all_label = self.all_label[_per, :]
        #     self.all_seq_len = self.all_seq_len[_per]

    # def _shuffle_data(self):
    #     _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
    #     self.all_data = self.all_data[_per, :, :]
    #     self.all_label = self.all_label[_per, :]
    #     self.all_seq_len = self.all_seq_len[_per]
    #
    # def next(self, batch_size, shuffle=True):
    #     if self.batch_id == len(self.all_seq_len):
    #         self.batch_id = 0
    #         if shuffle:
    #             self._shuffle_data()
    #     batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
    #     batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
    #     batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
    #     self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
    #     return batch_data, batch_labels, batch_seq_len


if __name__ == '__main__':
    train_sets = All_data_feature(foldname='../data/tmpdata/', max_seq=800,
                                num_class=10, trainable=True, kfold_num=4,
                                  feature_num=0)
    a, b, c = train_sets.next(1)
    print(a)
    print(b)
    print(c)