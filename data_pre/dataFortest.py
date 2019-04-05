'''
生成用于获得测试的数据
主要实现两个功能:
    1,根据文件夹获取需要测试的人的姓名
    2,根据文件夹和姓名获得对应人物所有数据用于测试
'''

import numpy as np
import os


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


def getAllPeople(foldname):
    '''
        根据文件夹获得所有需要测试的人,主要是根据文件夹获得文件夹下所有人名
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    return list(_person)