import os
import csv
import time
import shutil
import numpy as np
import pywt
import tensorflow as tf
import tmp_trans_wavelet

batch_size = 2
max_seq = 700
n_classes = 10
n_inputs = 8
n_hidden = 4

# 数据标准化方案1
def z_score(data_x):
    x_m = np.mean(data_x)
    x_p = np.std(data_x)
    x = (data_x-x_m)/x_p
    return x
def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix
def get_label(ch, num_classes=10):
    return np.eye(num_classes)[ch]
def get_lei(sq):
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    return alllei.index(sq)

class AllData_RNN(object):
    '''
    获得训练集与测试集数据,
    主要判断依据是trainable 与 kfold_num
    '''
    def __init__(self, foldname, max_seq=700, num_class=10):
        '''
        数据初始化函数,主要是生成RNN格式的训练数据与测试数据,判断依据是trainable与kfold_num
        :param foldname: 数据集合
        :param max_seq:
        :param num_class: 分类的数量
        :param trainable:
        :param kfold_num:
        '''
        self.all_data = []
        self.all_label = []
        self.all_seq_len = []
        self.batch_id = 0
        for filename in os.listdir(foldname):
            oa, ob, oc = filename.split('_')
            if oc == 'b.txt':
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


def LSTM_RNN(_X, seqlen, _weight, _bias):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cell_3 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3])
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # Get LSTM cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X, sequence_length=seqlen, dtype=tf.float32)
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    lstm_out = tf.reshape(tf.batch_gather(outputs, tf.to_int32(seqlen[:, None]-1)),[-1,n_hidden])
    # 方案二：
    # lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']

    # return tf.matmul(lstm_out, _weight['out']) + _bias['out']

def BiLSTM_RNN(_X, seqlen, _weight, _bias,):
    # shaping the dataSet
    # _X = tf.reshape(_X, [-1, n_inputs])
    # _X = tf.nn.relu(tf.matmul(_X, _weight['hidden']) + _bias['hidden'])
    # _X = tf.reshape(_X, [-1, max_seq, n_inputs])

    # net
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # backword
    lstm_cell_1_bw = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2_bw = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1_bw, lstm_cell_2_bw])
    # Get LSTM cell output
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cells_fw,
                                                 cell_bw=lstm_cells_bw,
                                                 inputs=_X,
                                                 sequence_length=tf.to_int32(seqlen),
                                                 dtype=tf.float32)
    _out1, _out2 = outputs
    # 方案一
    tf.reshape(tf.batch_gather(outputs, tf.to_int32(seqlen[:, None] - 1)), [-1, n_hidden])

    lstm_out_1 = tf.reshape(tf.batch_gather(_out1, tf.to_int32(seqlen[:, None] - 2)), [-1, n_hidden])
    lstm_out_2 = tf.reshape(tf.batch_gather(_out2, tf.to_int32(seqlen[:, None] - 2)), [-1, n_hidden])

    # 方案二
    # lstm_out_1 = tf.divide(tf.reduce_sum(_out1, 1), seqlen[:, None])
    # lstm_out_2 = tf.divide(tf.reduce_sum(_out2, 1), seqlen[:, None])
    return lstm_out_1, lstm_out_2

    # _out_last = lstm_out_1*0.7 + lstm_out_2*0.3
    # return tf.matmul(_out_last, _weight['out']) + _bias['out']


def main1():
    fold = './data/meanfilter_data/'
    tmp_trans_wavelet.main_datatrans(fold)

def main():
    dataRNN = AllData_RNN('./data/tmpdata/')
    batch_xs, batch_ys, batch_seq_len = dataRNN.next(batch_size)
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])

    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    #
    # weights = {
    #     'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),  # Hidden layer weights
    #     'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    # }
    # biases = {
    #     'hidden': tf.Variable(tf.random_normal([n_hidden])),
    #     'out': tf.Variable(tf.random_normal([n_classes]))
    # }

    pred = BiLSTM_RNN(x, seq_len, weights, biases)
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    res = sess.run(
        pred,
        feed_dict={
            x: batch_xs,
            y: batch_ys,
            seq_len: batch_seq_len
        }
    )
    print('res', res)

if __name__ == '__main__':
    main1()
