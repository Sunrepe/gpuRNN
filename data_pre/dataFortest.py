'''
生成用于获得测试的数据
主要实现两个功能:
    1,根据文件夹获取需要测试的人的姓名
    2,根据文件夹和姓名获得对应人物所有数据用于测试
'''

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
    x = (data_x - x_m) / x_p
    return x


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


class Per_CNNData(object):
    '''
    生成为RNN after CNN 的DataSet.
    注意,由于CNN 部分将原数据数据缩短了,所以生成数据的Length 是缩小了八倍的
    具体而言与CNN网络结构设置相关.

    细节:
        1,注意去除新数据中len>max_seq 的数据(该数据假设为动作分割不标准)
        2,注意使用了所有10类数据

    '''

    def __init__(self, foldname, person_name, max_seq=700, num_class=10):
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < num_class and oa==person_name:
                filenames = foldname + filename
                data = Read__mean_2(filenames)
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
                        self.all_seq_len.append(int(_len / 8.0) - 2)
                        s_tmp = np.zeros((max_seq, 10))
                        s_tmp[0:_len, 1:9] = tmp_data
                        s_tmp[:, 0] = s_tmp[:, 8]
                        s_tmp[:, 9] = s_tmp[:, 1]
                        self.all_data.append(s_tmp[:, :, np.newaxis])

        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')



class Per_RNNData(object):
    '''
    根据文件夹和人物名获得相关人物所有数据进行测试
    '''

    def __init__(self, foldname, person_name, max_seq=700, num_class=10):
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < num_class and oa == person_name:
                filenames = foldname + filename
                data = Read__mean_2(filenames)
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


class perAll_data_merge(object):
    '''
    获得9种特征用于最后测试
    '''
    def __init__(self, foldname, max_seq=800, num_class=10, testperson=None):
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
        __person = [testperson]
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
            if oc == 'b.txt' and oa in __person:
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
            self.data[i_toarray] = np.array(self.data[i_toarray]).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')