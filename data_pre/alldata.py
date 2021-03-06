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


class CNNData(object):
    '''
    生成为RNN after CNN 的DataSet.
    注意,由于CNN 部分将原数据数据缩短了,所以生成数据的Length 是缩小了八倍的
    具体而言与CNN网络结构设置相关.

    细节:
        1,注意去除新数据中len>max_seq 的数据(该数据假设为动作分割不标准)
        2,注意使用了所有10类数据

    '''

    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        __person = __person[0:1]
        # print('person:',__person)
        # __person = ['zhouxufeng']
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
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(int(_len/4.0)-2)
                        s_tmp = np.zeros((max_seq, 10))
                        s_tmp[0:_len, 1:9] = tmp_data
                        s_tmp[:, 0] = s_tmp[:, 8]
                        s_tmp[:, 9] = s_tmp[:, 1]
                        self.all_data.append(s_tmp[:, :, np.newaxis])

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')
        print('shape:',self.all_data.shape)
        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
            self.all_data = self.all_data[_per, :, :, :]
            self.all_label = self.all_label[_per, :]
            self.all_seq_len = self.all_seq_len[_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        self.all_data = self.all_data[_per, :, :]
        self.all_label = self.all_label[_per, :]
        self.all_seq_len = self.all_seq_len[_per]

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len


class RNNData(object):
    '''
    根据文件夹选择测试集与训练集

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
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
                    tmp_data = z_score(tmp_data)
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

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len

    def __len__(self):
        return len(self.all_label)


class fft1_RNNData(object):
    '''
    选择了所有的频率
    '''

    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        if not trainable : print(__person)
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
                    tmp_data = np.abs(np.fft.fft(tmp_data))
                    tmp_data = z_score(tmp_data)
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

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len

    def __len__(self):
        return len(self.all_label)


class fft2_RNNData(object):
    '''
    只保留采样率下有效频率:<=max_seq Hz
    '''

    def __init__(self, foldname, max_seq=100, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
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
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= 800:
                        pass
                    else:
                        tmp_data = np.abs(np.fft.fft(tmp_data))
                        tmp_data = z_score(tmp_data[0:max_seq, :])
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(max_seq)
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

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len

    def __len__(self):
        return len(self.all_label)


class fft3_RNNData(object):
    '''
    取得非对称区域的所有特征!.
    及除去第一个信号以外的所有信号的前一半
    '''

    def __init__(self, foldname, max_seq=400, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
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
                    _len = int(tmp_data.shape[0]/2)+1
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        tmp_data = np.abs(np.fft.fft(tmp_data))
                        tmp_data = z_score(tmp_data[0:_len, :])
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

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len

    def __len__(self):
        return len(self.all_label)


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
                    tmp_data = z_score(tmp_data)
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


class AllData_RNN_WAC(object):
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
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(_len-1)
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len-1, :] = tmp_data[0:-1,:]
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

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len


class AllDataold(object):
    '''
    根据文件夹获得数据，分为train/test
    func next(): 获得batch_data
    '''

    def __init__(self, foldname, max_seq=300, shuffle=True, num_class=8):
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < num_class:
                filenames = foldname + filename
                data = Read__mean_2(filenames)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                a__ = len(cutting)
                if len(oa) < 4:
                    a__ = 10
                # print("len:",a__, "\tfile:", filename)
                for cut in range(0, a__):
                    # 读取数据
                    if cut == 0:
                        tmp_data = z_score(data[0:cutting[cut], :])
                    else:
                        tmp_data = z_score(data[cutting[cut - 1]:cutting[cut], :])
                    # 生成数据
                    _len = tmp_data.shape[0]
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(_len)
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len] = tmp_data
                        self.all_data.append(s_tmp)
        # 打乱数据
        if shuffle:
            _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
            self.all_data = np.array(self.all_data)[_per, :, :].astype('float32')
            self.all_label = np.array(self.all_label)[_per, :].astype('float32')
            self.all_seq_len = np.array(self.all_seq_len)[_per].astype('float32')

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        self.all_data = np.array(self.all_data)[_per, :, :].astype('float32')
        self.all_label = np.array(self.all_label)[_per, :].astype('float32')
        self.all_seq_len = np.array(self.all_seq_len)[_per].astype('float32')

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


class waveandemg_RNNData(object):
    '''
    根据fold 和 kfold_num 获取wavelet 和 原始振幅信号 的DataSets

    细节:
        1,注意去除新数据中len>max_seq 的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        # train_person, test_person = getPersons(foldname, kfold_num)
        # __person = train_person if trainable else test_person
        __person = ['zhouxufeng']
        if not trainable:print(__person)
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        # wavelet
        self.all_data_wt = []
        self.all_label_wt = []
        self.all_seq_len_wt = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < num_class and oa in __person:
                # print(filename)
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, len(cutting)):
                    if cut == 0:
                        tmp_data = data[0:cutting[cut], :]
                    else:
                        tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                    tmp_data_wt = z_score(wavelet_trans(tmp_data))
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    _len_wt = tmp_data_wt.shape[0]
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
                        # wavelet
                        s_tmp_wt = np.zeros((max_seq, 8))
                        s_tmp_wt[0:_len_wt, :] = tmp_data_wt
                        self.all_data_wt.append(s_tmp_wt)

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')
        self.all_data_wt = np.array(self.all_data_wt).astype('float32')
        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
            self.all_data = self.all_data[_per, :, :]
            self.all_data_wt = self.all_data_wt[_per, :, :]
            self.all_label = self.all_label[_per, :]
            self.all_seq_len = self.all_seq_len[_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        self.all_data = self.all_data[_per, :, :]
        self.all_data_wt = self.all_data_wt[_per, :, :]
        self.all_label = self.all_label[_per, :]
        self.all_seq_len = self.all_seq_len[_per]

    def next(self, batch_size, shuffle=False):
        if self.batch_id == len(self.all_seq_len):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_data = self.all_data[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_seq_len = self.all_seq_len[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]
        batch_data_wt = self.all_data_wt[self.batch_id:min(self.batch_id + batch_size, len(self.all_seq_len))]

        self.batch_id = min(self.batch_id + batch_size, len(self.all_seq_len))
        return batch_data, batch_labels, batch_seq_len, batch_data_wt

    def __len__(self):
        return len(self.all_label)


class dwtData_RNN(object):
    '''
    获得dwt四层所有data
    并将四层融合使用于四层融合网络中
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
        # __person = ['zhouxufeng']
        if not trainable:print(__person)
        self.batch_id = 0  # use for batch_get
        self.all_label = []  # only one use
        tmp_use_len = [150,150,250,450]  # 用于生成指定格式长度的数据,对应于小波变换后4层系数最大长度
        self.data = {}
        self.seqlen = {}
        # 开始生产序列信号,存储于字典中
        for i_initdict in range(4):
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
                        coeffs = wavelet_trans_data(tmp_data)
                        for i_coeffs in range(4):
                            _len = coeffs[i_coeffs].shape[0]
                            self.seqlen[i_coeffs].append(_len)
                            s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
                            s_tmp[0:_len, :] = coeffs[i_coeffs]
                            self.data[i_coeffs].append(s_tmp)
        for i_toarray in range(4):
            self.seqlen[i_toarray] = np.array(self.seqlen[i_toarray]).astype('float32')
            self.data[i_toarray] = np.array(self.data[i_toarray]).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')

        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
            self.all_label = self.all_label[_per, :]
            for i_shuffle in range(4):
                self.data[i_shuffle] = self.data[i_shuffle][_per, :, :]
                self.seqlen[i_shuffle] = self.seqlen[i_shuffle][_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
        self.all_label = self.all_label[_per, :]
        for i_shuffle in range(4):
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
        for i_batch in range(4):
            batch_seq_len.append(self.seqlen[i_batch][self.batch_id:min(self.batch_id + batch_size, len(self.all_label))])
            batch_data.append(self.data[i_batch][self.batch_id:min(self.batch_id + batch_size, len(self.all_label))])
        self.batch_id = min(self.batch_id + batch_size, len(self.all_label))
        return batch_labels, batch_data, batch_seq_len


# class All_data_merge()
class All_data_merge(object):
    '''
    获得9种特征用于最后测试
    '''
    def __init__(self, foldname, max_seq=800, num_class=10, trainable=False, kfold_num=0):
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
        if not trainable:print(__person)
        self.batch_id = 0  # use for batch_get
        self.all_label = []  # only one use
        # 对应 dwt4层,ori,mean,std,zeros
        tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]
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
                        coeffs = wavelet_trans_data(tmp_data)
                        for i_coeffs in range(len(coeffs)):
                            _len = coeffs[i_coeffs].shape[0]
                            self.seqlen[i_coeffs].append(_len)
                            s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
                            s_tmp[0:_len, :] = coeffs[i_coeffs]
                            self.data[i_coeffs].append(s_tmp)
                        # 时域四层
                        coeffs = time_trans(tmp_data)
                        for i_coeffs in range(len(coeffs)):
                            _len = coeffs[i_coeffs].shape[0]
                            self.seqlen[i_coeffs+4].append(_len)
                            s_tmp = np.zeros((tmp_use_len[i_coeffs+4], 8))
                            s_tmp[0:_len, :] = coeffs[i_coeffs]
                            self.data[i_coeffs+4].append(s_tmp)

        for i_toarray in range(len(tmp_use_len)):
            self.seqlen[i_toarray] = np.array(self.seqlen[i_toarray]).astype('float32')
            # print(i_toarray)
            # for a in self.data[0]:
            #     print(a.shape)
            self.data[i_toarray] = np.array(self.data[i_toarray]).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')

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


class CNNData2(object):
    '''
    使用CNN网络进行特征提取
    注意其中使用卷积过程是长设置为1
    不进行通道融合。

    细节:
        1,注意去除新数据中len>max_seq 的数据(该数据假设为动作分割不标准)
        2,注意使用了所有10类数据

    '''

    def __init__(self, foldname, max_seq=700, num_class=10, trainable=False, kfold_num=0):
        train_person, test_person = getPersons(foldname, kfold_num)
        __person = train_person if trainable else test_person
        # __person = ['zhouxufeng']
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
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob), num_classes=num_class))
                        self.all_seq_len.append(get_last_dconvlen(_len))  # 计算经过CNN网络后的长度
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len, :] = tmp_data
                        self.all_data.append(s_tmp[:, :, np.newaxis])

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')
        # print('shape:',self.all_data.shape)
        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
            self.all_data = self.all_data[_per, :, :, :]
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


class All_data_merge_self(object):
    '''
    生成数据用于自身测试
    '''
    def __init__(self, foldname, max_seq=800, num_class=10, trainable=False, kfold_num=0):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        # train_person,test_person = getPersons(foldname, kfold_num)
        # __person = train_person if trainable else test_person
        # __person = ['zhouxufeng']
        # if not trainable:print(__person)
        self.batch_id = 0  # use for batch_get
        self.all_label = []  # only one use
        # 对应 dwt4层,ori,mean,std,zeros
        tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]
        self.data = {}
        self.seqlen = {}
        # 开始生产序列信号,存储于字典中
        for i_initdict in range(len(tmp_use_len)):
            self.data[i_initdict] = []
            self.seqlen[i_initdict] = []
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt':
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                _starts, _ends = data_len_get(len(cutting), kfold_num)  # 确定该区间范围
                # print('l:',_starts,'\tr:',_ends)
                for cut in range(0, len(cutting)):
                    # 根据是否可训练进行判断，选择
                    if trainable:
                        if cut<_starts or cut>=_ends:
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
                                coeffs = wavelet_trans_data(tmp_data)
                                for i_coeffs in range(len(coeffs)):
                                    _len = coeffs[i_coeffs].shape[0]
                                    self.seqlen[i_coeffs].append(_len)
                                    s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
                                    s_tmp[0:_len, :] = coeffs[i_coeffs]
                                    self.data[i_coeffs].append(s_tmp)
                                # 时域四层
                                coeffs = time_trans(tmp_data)
                                for i_coeffs in range(len(coeffs)):
                                    _len = coeffs[i_coeffs].shape[0]
                                    self.seqlen[i_coeffs + 4].append(_len)
                                    s_tmp = np.zeros((tmp_use_len[i_coeffs + 4], 8))
                                    s_tmp[0:_len, :] = coeffs[i_coeffs]
                                    self.data[i_coeffs + 4].append(s_tmp)
                        else:
                            pass
                    else:
                        if cut>=_starts and cut<_ends:
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
                                coeffs = wavelet_trans_data(tmp_data)
                                for i_coeffs in range(len(coeffs)):
                                    _len = coeffs[i_coeffs].shape[0]
                                    self.seqlen[i_coeffs].append(_len)
                                    s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
                                    s_tmp[0:_len, :] = coeffs[i_coeffs]
                                    self.data[i_coeffs].append(s_tmp)
                                # 时域四层
                                coeffs = time_trans(tmp_data)
                                for i_coeffs in range(len(coeffs)):
                                    _len = coeffs[i_coeffs].shape[0]
                                    self.seqlen[i_coeffs + 4].append(_len)
                                    s_tmp = np.zeros((tmp_use_len[i_coeffs + 4], 8))
                                    s_tmp[0:_len, :] = coeffs[i_coeffs]
                                    self.data[i_coeffs + 4].append(s_tmp)
                        else:
                            pass

        for i_toarray in range(len(tmp_use_len)):
            self.seqlen[i_toarray] = np.array(self.seqlen[i_toarray]).astype('float32')
            self.data[i_toarray] = np.array(self.data[i_toarray]).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')

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


class data_load_res(object):
    def __init__(self, foldname, trainable=False, kfold_num=0, fea_num=0):
        '''
        获得相关的中间过程。直接获得所有组合好的结果即可，不需要中间在训练过程中进行组合，加快训练速度。
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        # train_person,test_person = getPersons(foldname, kfold_num)
        # __person = train_person if trainable else test_person
        # __person = ['zhouxufeng']
        # if not trainable:print(__person)
        self.batch_id = 0  # use for batch_get
        if trainable:
            self.all_label = Read_data_res(foldname+'trainLabel_kfold{}'.format(kfold_num))  # only one use
            self.data_res = Read_data_res(foldname + 'train/fea{}_kfold{}'.format(0, kfold_num))
            for i_fea in range(1, 9):
                tmp_data = Read_data_res(foldname + 'train/fea{}_kfold{}'.format(i_fea, kfold_num))
                self.data_res = np.concatenate([self.data_res, tmp_data], 1)
        else:
            self.all_label = Read_data_res(foldname+'testLabel_kfold{}'.format(kfold_num))  # only one use
            self.data_res = Read_data_res(foldname + 'test/fea{}_kfold{}'.format(0, kfold_num))
            for i_fea in range(1, 9):
                tmp_data = Read_data_res(foldname + 'test/fea{}_kfold{}'.format(i_fea, kfold_num))
                self.data_res = np.concatenate([self.data_res, tmp_data], 1)

        # 打乱数据
        if trainable:
            _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
            self.all_label = self.all_label[_per, :]
            self.data_res = self.data_res[_per, :]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_label))  # 打乱后的行号
        self.all_label = self.all_label[_per, :]
        self.data_res = self.data_res[_per, :]

    def next(self, batch_size, shuffle=True):
        if self.batch_id == len(self.all_label):
            self.batch_id = 0
            if shuffle:
                self._shuffle_data()
        batch_labels = self.all_label[self.batch_id:min(self.batch_id + batch_size, len(self.all_label))]
        batch_data = self.data_res[self.batch_id:min(self.batch_id + batch_size, len(self.all_label))]
        return batch_labels, batch_data


class data_load_SVMfeas(object):
    def __init__(self, foldname, trainable=False, kfold_num=0):
        '''
        获得相关的中间过程。直接获得所有组合好的结果即可，不需要中间在训练过程中进行组合，加快训练速度。
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        train_person, test_person = getPersons_svm(foldname, kfold_num)
        __person = train_person if trainable else test_person
        print("All person:", __person)
        self.all_label = []
        self.data_res = []
        for pers in __person:
            data = Read_data_res("{}{}_svmfeas.txt".format(foldname, pers))
            for i in range(data.shape[0]):
                self.all_label.append(np.eye(10)[int(data[i, 32])])
                self.data_res.append(data[i, 0:32])
        self.all_label = np.array(self.all_label)
        self.data_res = np.array(self.data_res)


class data_load_SVMfeas_self(object):
    def __init__(self, foldname, trainable=False, kfold_num=0):
        '''
        获得相关的中间过程。直接获得所有组合好的结果即可，不需要中间在训练过程中进行组合，加快训练速度。
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        if trainable:
            _duan_name = 'train'
        else:
            _duan_name = 'test'
        self.all_label = []
        self.data_res = []
        data = Read_data_res("{}/{}_fold{}.txt".format(foldname, _duan_name, kfold_num))
        for i in range(data.shape[0]):
            self.all_label.append(np.eye(10)[int(data[i, 32])])
            self.data_res.append(data[i, 0:32])
        self.all_label = np.array(self.all_label)
        self.data_res = np.array(self.data_res)


class data_load_SVMfeas_self2(object):
    def __init__(self, foldname, trainable=False, kfold_num=0):
        '''
        获得相关的中间过程。直接获得所有组合好的结果即可，不需要中间在训练过程中进行组合，加快训练速度。
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        if trainable:
            _duan_name = 'train'
        else:
            _duan_name = 'test'
        self.all_label = []
        self.data_res = []
        data = Read_data_res("{}/mean_{}_fold{}.txt".format(foldname, _duan_name, kfold_num))
        for i in range(data.shape[0]):
            self.all_label.append(np.eye(10)[int(data[i, 8])])
            self.data_res.append(data[i, 0:8])
        self.all_label = np.array(self.all_label)
        self.data_res = np.array(self.data_res)


if __name__ == '__main__':
    train_sets = data_load_SVMfeas(foldname='../data/data_svmfea/', trainable=True, kfold_num=0)
    print(train_sets.data_res.shape)