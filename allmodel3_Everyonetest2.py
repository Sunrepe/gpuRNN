'''
使用Model3进行所有人群测试。
结果输出到针对每个个体，仅输出用于判断的算法的最终结果
与all_lstm1进行比较实验。
数据转存到data/res10/all_lstm3/中
注意需要保存标签和网络输出。
'''
# 用于测试新数据是否和老数据可以共用
from data_pre.dataFortest import *
import tensorflow as tf
import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800

k_fold_num = 1
model_name = 'models/all_lstm3/kfold{}/all_lstm3_model_kfold{}.ckpt'.format(k_fold_num, k_fold_num)

foldname = './data/actdata/'
matrix_save_path = "E:/Research-bachelor/storeMODELs/all_lstm_3/all_model3Matrix_kfold{}.txt".format(k_fold_num)

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def getPersons_every(foldname, kfold_num):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    _person = list(_person)
    _person.sort()
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    return test_p


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

    return lstm_out0+lstm_out1+lstm_out2+lstm_out3+lstm_out4+lstm_out5+lstm_out6+lstm_out7+lstm_out8


def main():
    time1 = time.time()

    # load sess and model
    # Graph input/output
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
    y = tf.placeholder(tf.float32, [None, n_classes])

    preds = LSTM_RNN_tmp(x0, x1, x2, x3, x4, x5, x6, x7, x8,
                         seq_len0, seq_len1, seq_len2, seq_len3,
                         seq_len4, seq_len5, seq_len6, seq_len7, seq_len8)
    # Loss, optimizer and evaluation
    saver = tf.train.Saver()
    # start train and test
    # To keep track of training's performance

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    assert os.path.exists(foldname)
    saver.restore(sess, model_name)
    # load accomplished
    print('Model are loaded!\t{}s'.format(time.time()-time1))
    # pathss = saver.save(sess, 'models/all_lstm3/kfold{}/all_lstm3_model_kfold{}.ckpt'.format(k_fold_num, k_fold_num))
    # print("Model resaved at {}".format(pathss))
    # df_data = []
    print("Start test!")
    person_list = getPersons_every(foldname, k_fold_num)

    for person in person_list:
        time2 = time.time()
        print(person)
        test_sets = perAll_data_merge(foldname=foldname,
                                      max_seq=max_seq,
                                      num_class=n_classes,
                                      testperson=person)
        result_labels = test_sets.all_label
        feed_dic = {
            y: test_sets.all_label,
            x0: test_sets.data[0],
            x1: test_sets.data[1],
            x2: test_sets.data[2],
            x3: test_sets.data[3],
            x4: test_sets.data[4],
            x5: test_sets.data[5],
            x6: test_sets.data[6],
            x7: test_sets.data[7],
            x8: test_sets.data[8],
            seq_len0: test_sets.seqlen[0],
            seq_len1: test_sets.seqlen[1],
            seq_len2: test_sets.seqlen[2],
            seq_len3: test_sets.seqlen[3],
            seq_len4: test_sets.seqlen[4],
            seq_len5: test_sets.seqlen[5],
            seq_len6: test_sets.seqlen[6],
            seq_len7: test_sets.seqlen[7],
            seq_len8: test_sets.seqlen[8]
        }
        # Accuracy for test data
        one_hot_predictions = sess.run(
            preds,
            feed_dict=feed_dic
        )
        Matrix_to_CSV('data/res10/all_lstm3/pre_{}'.format(person), one_hot_predictions)
        Matrix_to_CSV('data/res10/all_lstm3/label_{}'.format(person), result_labels)
        print("finish time: {}\n".format(time.time()-time2))
    sess.close()


if __name__ == '__main__':
    main()