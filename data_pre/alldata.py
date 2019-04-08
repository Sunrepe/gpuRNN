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
    _person.remove('marui')
    _person.remove('zhangyixuan')
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    train_p = ['marui', 'zhangyixuan']
    for i in _person:
        if i not in test_p:
            train_p.append(i)
    return train_p,test_p


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


class NewDataSetTest(object):
    '''
    对于新数据进行测试的第一个数据类:与原数据同步,用原数据测试新数据
    之后还会有NewDataSetTest2等,进行单独的新数据测试,或者进行新老数据混合训练集测试集测试

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=300, shuffle=False):

        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < 8:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 20):
                    if cut == 0:
                        tmp_data = z_score(data[0:cutting[cut], :])
                    else:
                        tmp_data = z_score(data[cutting[cut - 1]:cutting[cut], :])
                    _per = [i for i in range(0, tmp_data.shape[0], 3)]
                    tmp_data = tmp_data[_per, :]
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if not _len >= 200:
                        # 生成数据
                        self.all_label.append(get_label(get_lei(ob)))
                        self.all_seq_len.append(_len)
                        s_tmp = np.zeros((max_seq, 8))
                        s_tmp[0:_len] = tmp_data
                        self.all_data.append(s_tmp)
        self.all_data = np.array(self.all_data).astype('float32')
        self.all_label = np.array(self.all_label).astype('float32')
        self.all_seq_len = np.array(self.all_seq_len).astype('float32')
        # 打乱数据
        if shuffle:
            _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
            self.all_data = self.all_data[_per, :, :]
            self.all_label = self.all_label[_per, :]
            self.all_seq_len = self.all_seq_len[_per]

    def _shuffle_data(self):
        _per = np.random.permutation(len(self.all_seq_len))  # 打乱后的行号
        self.all_data = np.array(self.all_data)[_per, :, :].astype('float32')
        self.all_label = np.array(self.all_label)[_per, :].astype('float32')
        self.all_seq_len = np.array(self.all_seq_len)[_per].astype('float32')

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


class NewDataSetTest2(object):
    '''
    对于新数据进行测试2:只考虑新数据,分为训练集和测试集(少量)
    之后还会有NewDataSetTest2等,进行单独的新数据测试,或者进行新老数据混合训练集测试集测试

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=300, shuffle=True, trainable=True):
        # allmen = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'gaoyan', 'xiejiabao']
        train_person = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'simengbin']
        test_person = ['gaoyan', 'xiejiabao']
        self.__people = train_person if trainable else test_person
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < 10 and oa in self.__people:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 20):
                    if cut == 0:
                        tmp_data = z_score(data[0:cutting[cut], :])
                    else:
                        tmp_data = z_score(data[cutting[cut - 1]:cutting[cut], :])
                    _per = [i for i in range(0, tmp_data.shape[0], 2)]
                    tmp_data = tmp_data[_per, :]
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= 600:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(ob, num_classes=10))
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


class NewDataSetTest3(object):
    '''
    对于新数据进行测试2:只考虑新数据,分为训练集和测试集(少量)
    注意:原始数据帧率较高,只选择其1/4帧率进行测试
    之后还会有NewDataSetTest2等,进行单独的新数据测试,或者进行新老数据混合训练集测试集测试

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=300, shuffle=True, trainable=True):
        # allmen = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'gaoyan', 'xiejiabao']
        train_person = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'simengbin']
        test_person = ['gaoyan', 'xiejiabao']
        self.__people = train_person if trainable else test_person
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < 10 and oa in self.__people:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 20):
                    if cut == 0:
                        tmp_data = data[0:cutting[cut], :]
                    else:
                        tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                    _per = [i for i in range(0, tmp_data.shape[0], 2)]
                    tmp_data = tmp_data[_per, :]
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(ob, num_classes=10))
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


class NewDataSetTest4(object):
    '''
    对于新数据进行测试2:只考虑新数据,分为训练集和测试集(少量)
    注意:原始数据帧率较高,只选择其1/4帧率进行测试
    之后还会有NewDataSetTest2等,进行单独的新数据测试,或者进行新老数据混合训练集测试集测试

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=300, shuffle=True, trainable=True):
        # allmen = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'gaoyan', 'xiejiabao']
        train_person = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'simengbin']
        test_person = ['gaoyan', 'xiejiabao']
        self.__people = train_person if trainable else test_person
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_lei(ob) < 10 and oa in self.__people:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 20):
                    if cut == 0:
                        tmp_data = data[0:cutting[cut], :]
                    else:
                        tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                    _per = [i for i in range(0, tmp_data.shape[0], 4)]
                    tmp_data = tmp_data[_per, :]
                    tmp_data = z_score(tmp_data)
                    _len = tmp_data.shape[0]
                    # 读取数据
                    if _len >= max_seq:
                        pass
                    else:
                        # 生成数据
                        self.all_label.append(get_label(ob, num_classes=10))
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

