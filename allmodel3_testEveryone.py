'''
使用Model3进行所有人群测试。
一共进行五次，存在5折交叉验证。
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

k_fold_num = 4
model_name = "E:/Research-bachelor/storeMODELs/all_lstm_3/all_model3_kfold{}/" \
             "model_mergeall_kfold4.ckpt-3600".format(k_fold_num,k_fold_num)

foldname = './data/actdata/'
matrix_save_path = "E:/Research-bachelor/storeMODELs/all_lstm_3/all_model3Matrix_kfold{}.txt".format(k_fold_num)

df_save_path = 'E:/Research-bachelor/storeMODELs/all_lstm_3/allmodel3_everyone_precision_kfold{}.csv'.format(k_fold_num)
df_save_path2 = 'E:/Research-bachelor/storeMODELs/all_lstm_3/allmodel3_everyone_recall_kfold{}.csv'.format(k_fold_num)
df_save_path3 = 'E:/Research-bachelor/storeMODELs/all_lstm_3/allmodel3_everyone_f1score_kfold{}.csv'.format(k_fold_num)

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

    return [lstm_out0,lstm_out1,lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8]
    # return tf.matmul(lstm_out, _weight['out']) + _bias['out']


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

    # df_data = []
    print("Start test!")
    person_list = getPersons_every(foldname, k_fold_num)
    print("persons:", person_list)
    every_stream_matrix = {}
    for i_stram in range(10):
        every_stream_matrix[i_stram] = np.zeros([10,10])
    every_stream_precision = {}
    for i_stram in range(10):
        every_stream_precision[i_stram] = []
    every_stream_recall = {}
    for i_stram in range(10):
        every_stream_recall[i_stram] = []
    every_stream_f1score = {}
    for i_stram in range(10):
        every_stream_f1score[i_stram] = []
    # every_precision = {1:[],2:[],3:[]}
    # every_recall = {1:[],2:[],3:[]}
    # every_f1score = {1:[],2:[],3:[]}
    for person in person_list:
        print(person)
        test_sets = perAll_data_merge(foldname=foldname,
                                      max_seq=max_seq,
                                      num_class=n_classes,
                                      testperson=person)

        result_labels = test_sets.all_label.argmax(1)
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
        for i_pre in range(9):
            pre_onehot = one_hot_predictions[i_pre]
            predictions = pre_onehot.argmax(1)
            _percision = metrics.precision_score(result_labels, predictions, average="weighted")
            _recall = metrics.recall_score(result_labels, predictions, average="weighted")
            _f1Score = metrics.f1_score(result_labels, predictions, average="weighted")
            every_stream_precision[i_pre].append(_percision)
            every_stream_recall[i_pre].append(_recall)
            every_stream_f1score[i_pre].append(_f1Score)
            # matrix
            confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
            # print(confusion_matrix)
            every_stream_matrix[i_pre] += confusion_matrix

        predictions = sum(one_hot_predictions).argmax(1)
        _percision = metrics.precision_score(result_labels, predictions, average="weighted")
        _recall = metrics.recall_score(result_labels, predictions, average="weighted")
        _f1Score = metrics.f1_score(result_labels, predictions, average="weighted")
        # zhengcun
        every_stream_precision[9].append(_percision)
        every_stream_recall[9].append(_recall)
        every_stream_f1score[9].append(_f1Score)

        print("Precision: {}%".format(100 * _percision))
        print("Recall: {}%".format(100 * _recall))
        print("f1_score: {}%".format(100 * _f1Score))
        # print("Precision: ",metrics.precision_score(result_labels, predictions, average=None))
        # # print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions)))
        # # print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions)))

        print("{}'s Confusion Matrix:".format(person))
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        print(confusion_matrix)
        every_stream_matrix[9] += confusion_matrix
    print("All stream matrix:")
    for i_st in range(10):
        print(every_stream_matrix[i_st])
        every_stream_matrix[i_st] = every_stream_matrix[i_st].astype('int32')
        Matrix_to_CSV(matrix_save_path, every_stream_matrix[i_st])
    # person_list.append('allperson_kfold{}'.format(k_fold_num))
    # df = pd.DataFrame(every_stream_precision[0], index=person_list, columns=["S"])
    df1 = pd.DataFrame(every_stream_precision, index=person_list)
    df2 = pd.DataFrame(every_stream_recall, index=person_list)
    df3 = pd.DataFrame(every_stream_f1score, index=person_list)
    # df = pd.DataFrame(df_data, index=person_list, columns=["Recall","Precision","F1_score","Total_wrong"])

    df1.to_csv(df_save_path)
    df2.to_csv(df_save_path2)
    df3.to_csv(df_save_path3)

    sess.close()


if __name__ == '__main__':
    main()