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
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no']
    return alllei.index(sq)


def get_label(ch, num_classes=8):
    return np.eye(num_classes)[get_lei(ch)]


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

    def __init__(self, foldname, max_seq=300, shuffle=True):

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
                for cut in range(0, 10):
                    # 读取数据
                    if cut == 0:
                        tmp_data = z_score(data[0:cutting[cut] - 1, :])
                        _len = cutting[0]
                    else:
                        tmp_data = z_score(data[cutting[cut - 1]:cutting[cut] - 1, :])
                        _len = cutting[cut] - cutting[cut - 1]
                    # 生成数据
                    self.all_label.append(get_label(ob))
                    self.all_seq_len.append(_len)
                    s_tmp = np.zeros((max_seq, 8))
                    # print(np.shape(tmp_data))
                    # print('len_', _len)
                    s_tmp[0:_len-1] = tmp_data
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

