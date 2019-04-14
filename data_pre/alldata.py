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
    _person.remove('marui')
    _person.remove('zhangyixuan')
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    train_p = ['marui', 'zhangyixuan']
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