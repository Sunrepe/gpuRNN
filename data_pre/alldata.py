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
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'snap', 'no', 'finger']
    return alllei.index(sq)


def get_8class2(sq):
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'snap', 'finger']
    return alllei.index(sq)


def get_label(ch, num_classes=8):
    return np.eye(num_classes)[ch]


# 数据标准化方案1
def z_score(data_x):
    x_m = np.mean(data_x)
    x_p = np.std(data_x)
    x = (data_x-x_m)/x_p
    return x


class AllData(object):
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
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 10):
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


class NewData8class1(object):
    '''
    对于新数据进行测试4:只选择其中8个动作进行分类
    注意:原始数据帧率较高,只选择其1/4帧率进行测试
    之后还会有NewDataSetTest2等,进行单独的新数据测试,或者进行新老数据混合训练集测试集测试

    细节:
        1,注意去除新数据中len>600的数据(该数据假设为动作分割不标准)
        2,注意只使用前8类

    '''

    def __init__(self, foldname, max_seq=300, shuffle=True, trainable=True, num_class=8):
        # allmen = ['zhangqijian', 'wupanhao', 'zhouxufeng', 'wangzihan', 'wanyuanqiang', 'shechen', 'gaoyan', 'xiejiabao']
        train_person = ['zhangqijian', 'gaoyan', 'wangzihan', 'wanyuanqiang', 'shechen', 'simengbin', 'xiejiabao']
        test_person = ['zhouxufeng', 'wupanhao']
        self.__people = train_person if trainable else test_person
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt' and get_8class(ob) < 8 and oa in self.__people:
                filename = foldname + filename
                data = Read__mean_2(filename)
                cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
                for cut in range(0, 20):
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
                        self.all_label.append(get_label(get_8class(ob), num_classes=num_class))
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
