'''
实时获取数据并识别！
'''
from data_pre.diffeature import *
import tensorflow as tf
import collections
import myo
import time
import sys
import csv
import os
import numpy as np

# all pose:use it for name output_file

alllei = ['p', 'double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']

# Global parameters:
motion_Label = 10
num_act = 20
windows_len = 80
threshold = 5.5

n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800
tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]

# Training
# learning_rate = 0.0001
lambda_loss_amount = 0.0040
training_iters = 500  # Loop 1000 times on the dataset
batch_size = 400
display_iter = 4000  # To show test set accuracy during training
model_save = 20

k_fold_num = 2
model_name = 'models/all_lstm3/kfold{}/all_lstm3_model_kfold{}.ckpt'.format(k_fold_num, k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow([row])


def data_precess(tmp_data):
    # 初始化
    tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]
    data = {}
    seqlen = {}
    # 开始生产序列信号,存储于字典中
    for i_initdict in range(len(tmp_use_len)):
        data[i_initdict] = []
        seqlen[i_initdict] = []
    # 开始记录
    _len = tmp_data.shape[0]
    if _len >= max_seq:
        pass
    else:
        # 生成数据
        coeffs = wavelet_trans_data(tmp_data)
        for i_coeffs in range(len(coeffs)):
            _len = coeffs[i_coeffs].shape[0]
            seqlen[i_coeffs].append(_len)
            s_tmp = np.zeros((tmp_use_len[i_coeffs], 8))
            s_tmp[0:_len, :] = coeffs[i_coeffs]
            data[i_coeffs].append(s_tmp)
        # 时域四层
        coeffs = time_trans(tmp_data)
        for i_coeffs in range(len(coeffs)):
            _len = coeffs[i_coeffs].shape[0]
            seqlen[i_coeffs + 4].append(_len)
            s_tmp = np.zeros((tmp_use_len[i_coeffs + 4], 8))
            s_tmp[0:_len, :] = coeffs[i_coeffs]
            data[i_coeffs + 4].append(s_tmp)

    for i_toarray in range(len(tmp_use_len)):
        seqlen[i_toarray] = np.array(seqlen[i_toarray]).astype('float32')
        data[i_toarray] = np.array(data[i_toarray]).astype('float32')
    return data, seqlen


def LSTM_RNN_tmp(x0,x1,x2,x3,x4,x5,x6,x7,x8,
                 seq0,seq1,seq2,seq3,seq4,seq5,seq6,seq7,seq8):
    # dwt
    with tf.variable_scope('dwt0'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x0, sequence_length=tf.to_int32(seq0), dtype=tf.float32)
        lstm_out0 = tf.divide(tf.reduce_sum(outputs, 1), seq0[:, None])
        lstm_out0 = tf.nn.dropout(lstm_out0, keep_prob=0.8)
        lstm_out0 = tf.layers.dense(lstm_out0, 10)
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x1, sequence_length=tf.to_int32(seq1), dtype=tf.float32)
        lstm_out1 = tf.divide(tf.reduce_sum(outputs, 1), seq1[:, None])
        lstm_out1 = tf.nn.dropout(lstm_out1, keep_prob=0.8)
        lstm_out1 = tf.layers.dense(lstm_out1, 10)
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x2, sequence_length=tf.to_int32(seq2), dtype=tf.float32)
        lstm_out2 = tf.divide(tf.reduce_sum(outputs, 1), seq2[:, None])
        lstm_out2 = tf.nn.dropout(lstm_out2, keep_prob=0.5)
        lstm_out2 = tf.layers.dense(lstm_out2, 10)
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x3, sequence_length=tf.to_int32(seq3), dtype=tf.float32)
        lstm_out3 = tf.divide(tf.reduce_sum(outputs, 1), seq3[:, None])
        lstm_out3 = tf.nn.dropout(lstm_out3, keep_prob=0.5)
        lstm_out3 = tf.layers.dense(lstm_out3, 10)

    # time domain
    with tf.variable_scope('ori'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x4, sequence_length=tf.to_int32(seq4), dtype=tf.float32)
        lstm_out4 = tf.divide(tf.reduce_sum(outputs, 1), seq4[:, None])
        lstm_out4 = tf.nn.dropout(lstm_out4, keep_prob=0.8)
        lstm_out4 = tf.layers.dense(lstm_out4, 10)
    with tf.variable_scope('mean'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x5, sequence_length=tf.to_int32(seq5), dtype=tf.float32)
        lstm_out5 = tf.divide(tf.reduce_sum(outputs, 1), seq5[:, None])
        lstm_out5 = tf.nn.dropout(lstm_out5, keep_prob=0.8)
        lstm_out5 = tf.layers.dense(lstm_out5, 10)
    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x6, sequence_length=tf.to_int32(seq6), dtype=tf.float32)
        lstm_out6 = tf.divide(tf.reduce_sum(outputs, 1), seq6[:, None])
        lstm_out6 = tf.nn.dropout(lstm_out6, keep_prob=0.6)
        lstm_out6 = tf.layers.dense(lstm_out6, 10)
    with tf.variable_scope('wtchange'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x7, sequence_length=tf.to_int32(seq7), dtype=tf.float32)
        lstm_out7 = tf.divide(tf.reduce_sum(outputs, 1), seq7[:, None])
        lstm_out7 = tf.nn.dropout(lstm_out7, keep_prob=0.5)
        lstm_out7 = tf.layers.dense(lstm_out7, 10)
    with tf.variable_scope('fft'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x8, sequence_length=tf.to_int32(seq8), dtype=tf.float32)
        lstm_out8 = tf.divide(tf.reduce_sum(outputs, 1), seq8[:, None])
        lstm_out8 = tf.nn.dropout(lstm_out8, keep_prob=0.5)
        lstm_out8 = tf.layers.dense(lstm_out8, 10)

    return [lstm_out0,lstm_out1,lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8]
    # return tf.matmul(lstm_out, _weight['out']) + _bias['out']


class EmgDataRecode(myo.DeviceListener):
    def __init__(self, n_Windows, f_forcut):
        super(EmgDataRecode, self).__init__()
        # time/last_time/n for rate recode
        self.__slideWindow = collections.deque(maxlen=n_Windows)
        self.__nfwindows = float(n_Windows)
        self.__fcut = f_forcut
        self.activeEMG = collections.deque()  # 记录活跃数据
        self.active_NUM = 0

        self.data_for_pre = np.zeros([10,10])  # 单独记录用于每次进行动作判断
        self.time_for_pre_start = 0  # 配合使用，记录动作完成时间。

        self.onrecdoe = False
        self.unrelax = False
        self.tmpslide = 0.0  # just for print check:see the
        self.tmplen = 0  # just for print check

        self.__behind = int(n_Windows/2)
        self.__allwait = False
        self.__tt = None

    def on_connected(self, event):
        print("Hello, '{}'! Double tap to exit.".format(event.device_name))
        event.device.stream_emg(True)

    def calulate_slideWindows(self):
        p = []
        for _ in self.__slideWindow:
            p.append(sum(list(map(abs, _)))/8.0)
        return sum(p)/self.__nfwindows

    def on_emg(self, event):
        self.__emg = event.emg
        self.__slideWindow.append(self.__emg)

        self.tmpslide = self.calulate_slideWindows()
        self.onrecdoe = True if(self.tmpslide > self.__fcut) else False

        # 根据状态进行记录
        # active结束了,但是unrelax还未结束,表示该动作截止.
        if self.__allwait:
            if self.__behind == 1:
                self.__allwait = False
                self.time_for_pre_start = time.clock()
                print()
                print('Valid Act_Length:', len(self.activeEMG))
                print("Valid EMG_Rate:", len(self.activeEMG)/(time.clock()-self.__tt))
                self.active_NUM += 1
                self.data_for_pre = np.array(self.activeEMG)
                self.activeEMG.clear()
                self.__behind = int(self.__nfwindows/2.0)
            else:
                self.__behind -= 1
                self.activeEMG.append(self.__emg)
        else:
            if self.onrecdoe:
                if not self.unrelax:
                    self.__tt = time.clock()
                    for _ in range(int(self.__nfwindows/2), int(self.__nfwindows)):
                        self.activeEMG.append(self.__slideWindow[_])
                else:
                    self.activeEMG.append(self.__emg)
            elif self.unrelax:  # 记录结束,写入结果并及时改写unrelax和清空activeEMG
                self.unrelax = False
                print()
                print('Act length:', len(self.activeEMG), end='')
                # sys.stdout.flush()
                if len(self.activeEMG) > 150:
                    self.__allwait = True
                else:
                    self.activeEMG.clear()
                    self.__tt = None
            self.unrelax = True if self.onrecdoe else False


def main():
    # loading models -------------
    # Graph input/output
    time1 = time.time()
    total_time = []
    total_time_2 = []
    x0 = tf.placeholder(tf.float32, [None, tmp_use_len[0], n_inputs])
    x1 = tf.placeholder(tf.float32, [None, tmp_use_len[1], n_inputs])
    x2 = tf.placeholder(tf.float32, [None, tmp_use_len[2], n_inputs])
    x3 = tf.placeholder(tf.float32, [None, tmp_use_len[3], n_inputs])
    x4 = tf.placeholder(tf.float32, [None, tmp_use_len[4], n_inputs])
    x5 = tf.placeholder(tf.float32, [None, tmp_use_len[5], n_inputs])
    x6 = tf.placeholder(tf.float32, [None, tmp_use_len[6], n_inputs])
    x7 = tf.placeholder(tf.float32, [None, tmp_use_len[7], n_inputs])
    x8 = tf.placeholder(tf.float32, [None, tmp_use_len[8], n_inputs])
    seq_len0 = tf.placeholder(tf.float32, [None])
    seq_len1 = tf.placeholder(tf.float32, [None])
    seq_len2 = tf.placeholder(tf.float32, [None])
    seq_len3 = tf.placeholder(tf.float32, [None])
    seq_len4 = tf.placeholder(tf.float32, [None])
    seq_len5 = tf.placeholder(tf.float32, [None])
    seq_len6 = tf.placeholder(tf.float32, [None])
    seq_len7 = tf.placeholder(tf.float32, [None])
    seq_len8 = tf.placeholder(tf.float32, [None])

    preds = LSTM_RNN_tmp(x0, x1, x2, x3, x4, x5, x6, x7, x8,
                         seq_len0, seq_len1, seq_len2, seq_len3,
                         seq_len4, seq_len5, seq_len6, seq_len7, seq_len8)
    pred = sum(preds)
    saver = tf.train.Saver(max_to_keep=12)
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    saver.restore(sess, model_name)

    print('Model are loaded!\t{}s'.format(time.time()-time1))

    # myo 处理
    myo.init(sdk_path='./data_pre/myo-sdk-win-0.9.0/')
    hub = myo.Hub()
    listener = EmgDataRecode(n_Windows=windows_len, f_forcut=threshold)
    last_pose_n = 0
    while hub.run(listener.on_event, 500):
        print('\rAct Num:', listener.active_NUM, '\tSlide mean:', listener.tmpslide, end='')
        sys.stdout.flush()
        if listener.active_NUM >= num_act:
            break
        if listener.active_NUM > last_pose_n:
            # 此处执行判断操作
            # print(listener.data_for_pre.shape)
            time2 = time.clock()
            data_, seqlen_ = data_precess(listener.data_for_pre)
            feed_dic = {
                x0: data_[0],
                x1: data_[1],
                x2: data_[2],
                x3: data_[3],
                x4: data_[4],
                x5: data_[5],
                x6: data_[6],
                x7: data_[7],
                x8: data_[8],
                seq_len0: seqlen_[0],
                seq_len1: seqlen_[1],
                seq_len2: seqlen_[2],
                seq_len3: seqlen_[3],
                seq_len4: seqlen_[4],
                seq_len5: seqlen_[5],
                seq_len6: seqlen_[6],
                seq_len7: seqlen_[7],
                seq_len8: seqlen_[8]
            }
            # Accuracy for test data
            one_hot_predictions = sess.run(
                pred,
                feed_dict=feed_dic
            )
            total_time.append(time.clock()-listener.time_for_pre_start)
            total_time_2.append(time.clock()-time2)
            print("All time for pre is {}s".format(time.time()-listener.time_for_pre_start))
            print(one_hot_predictions)
            print("Pred act is: {}".format(one_hot_predictions.argmax(1)))
            print("Pred act is: {}".format(LABELS[one_hot_predictions.argmax(1)[0]]))
            last_pose_n += 1
            pass
    Matrix_to_CSV('total1.csv', total_time)
    Matrix_to_CSV('total2.csv', total_time_2)
    print("\nYour Pose:", alllei[motion_Label])
    print("\n\033[1;32;mFinish!  Please have a rest!")

    # pic for check!

if __name__ == '__main__':
    main()
