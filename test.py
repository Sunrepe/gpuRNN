'''
使用Model3进行所有人群测试。
一共进行五次，存在5折交叉验证。
'''
# 用于测试新数据是否和老数据可以共用
from data_pre.alldata import *
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
# model_name = 'E:/Research-bachelor/storeMODELs/5kfold/lstmkfold{}/lstm/model_LSTM_kfold{}.ckpt-final'.format(k_fold_num,k_fold_num)

model_name = "E:/Research-bachelor/storeMODELs/all_lstm_3/all_model3_kfold{}/" \
             "model_mergeall_kfold4.ckpt-3600".format(k_fold_num,k_fold_num)
fold = './data/actdata/'
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

def LSTM_RNN_tmp(x0,x1,x2,seq0,seq1,seq2):
    with tf.variable_scope('ori'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x0, sequence_length=tf.to_int32(seq0), dtype=tf.float32)
        lstm_out0 = tf.divide(tf.reduce_sum(outputs, 1), seq0[:, None])

    with tf.variable_scope('avg'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x1, sequence_length=tf.to_int32(seq1), dtype=tf.float32)
        lstm_out1 = tf.divide(tf.reduce_sum(outputs, 1), seq1[:, None])

    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x2, sequence_length=tf.to_int32(seq2), dtype=tf.float32)
        lstm_out2 = tf.divide(tf.reduce_sum(outputs, 1), seq2[:, None])

    lstm_out0 = tf.concat([lstm_out0, lstm_out1], 1)
    lstm_out0 = tf.concat([lstm_out0, lstm_out2], 1)
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out0, 10)
    return lstm_out




def main():
    time1 = time.time()

    # load sess and model
    # Graph input/output
    x0 = tf.placeholder(tf.float32, [None, tmp_use_len[0], n_inputs])
    x1 = tf.placeholder(tf.float32, [None, tmp_use_len[1], n_inputs])
    x2 = tf.placeholder(tf.float32, [None, tmp_use_len[2], n_inputs])
    seq_len0 = tf.placeholder(tf.float32, [None])
    seq_len1 = tf.placeholder(tf.float32, [None])
    seq_len2 = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.float32, [None, n_classes])

    preds = LSTM_RNN_tmp(x0, x1, x2, seq_len0, seq_len1, seq_len2)
    # Loss, optimizer and evaluation
    saver = tf.train.Saver()
    # start train and test
    # To keep track of training's performance



    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    # assert os.path.exists(foldname)
    saver.restore(sess, model_name)
    # load accomplished
    print('Model are loaded!\t{}s'.format(time.time()-time1))

    var = tf.global_variables()  # 取出全局中所有的参数
    var_flow_restore2 = [val for val in var if 'avg' in val.name]  # 取出名字中有‘flownet’的参数
    saver2 = tf.train.Saver(var_flow_restore2)  # 这句话就是关键了，可以网Saver中传参数
    saver2.restore(sess, './tmpmodel/1/model_feature1_kfold0.ckpt-3200')  # 然后就往sess对应的图中导入了参数（var_flow_restore

    # df_data = []
    print("Start test!")
    test_sets = All_data_merge(foldname=fold, max_seq=max_seq,
                               num_class=n_classes, trainable=False, kfold_num=k_fold_num)

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

    predictions = sum(one_hot_predictions).argmax(1)
    _percision = metrics.precision_score(result_labels, predictions, average="weighted")
    _recall = metrics.recall_score(result_labels, predictions, average="weighted")
    _f1Score = metrics.f1_score(result_labels, predictions, average="weighted")
    # zhengcun

    print("Precision: {}%".format(100 * _percision))
    print("Recall: {}%".format(100 * _recall))
    print("f1_score: {}%".format(100 * _f1Score))

    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    print(confusion_matrix)


    sess.close()


if __name__ == '__main__':
    main()