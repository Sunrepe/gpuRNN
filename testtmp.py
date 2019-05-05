'''
根据文件夹生成相关文件。
此处作用是，根据训练好的Model，生成可以后期使用的res10与Res50数据
'''

from data_pre.diffeature import *
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warnings and errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800
tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]

# Training
# learning_rate = 0.0001
lambda_loss_amount = 0.0020
training_iters = 200  # Loop 1000 times on the dataset
batch_size = 400
display_iter = 4000  # To show test set accuracy during training
model_save = 20

fold = './data/actdata/'
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def get_model(fea_num, kfold_num):
    pass
    if fea_num == 0 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 1 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 2 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 3 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2400'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 4 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2400'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 5 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2600'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 6 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 7 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 8 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 0 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 1 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 2 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 3 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    return model_path


def LSTM_RNN_f0(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('ori'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f1(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('avg'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f2(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f3(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('wlc'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f4(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f5(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f6(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f7(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt4'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f8(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('fft'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def main():
    # time0 = time.time()

    time1 = time.time()
    # # Graph weights
    # with tf.variable_scope("weight"):
    #     weights = {
    #         'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    #     }
    #     biases = {
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    # #
    # # # Graph input/output
    # x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    # y = tf.placeholder(tf.float32, [None, n_classes])
    # seq_len = tf.placeholder(tf.float32, [None])
    #
    k_fold_num = 2
    feature_num__s = 0
    # pred = LSTM_RNN_f5(x, seq_len, weights, biases)
    # #
    # with tf.name_scope('fullConnect'):
    #     lstm_out = tf.matmul(pred, weights['out']) + biases['out']
    #
    # correct_pred = tf.equal(tf.argmax(lstm_out, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # saver = tf.train.Saver(max_to_keep=12)

    print('loading data...')
    train_sets = All_data_feature_lable(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=True, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    test_sets = All_data_feature_lable(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=False, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time() - time1)

    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    # saver.restore(sess, "./models/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s, k_fold_num))

    # save 中间结果
    # res50, res10 = sess.run(
    #     [pred, lstm_out],
    #     feed_dict={
    #         x: train_sets.all_data,
    #         y: train_sets.all_label,
    #         seq_len: train_sets.all_seq_len
    #     }
    # )
    # Matrix_to_CSV('./data/res50/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    # Matrix_to_CSV('./data/res10/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)

    # res50, res10 = sess.run(
    #     [pred, lstm_out],
    #     feed_dict={
    #         x: test_sets.all_data,
    #         y: test_sets.all_label,
    #         seq_len: test_sets.all_seq_len
    #     }
    # )
    # Matrix_to_CSV('./data/res50/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    # Matrix_to_CSV('./data/res10/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)

    # label
    if feature_num__s == 0:
        # train
        print('kfold:{}'.format(k_fold_num))
        Matrix_to_CSV('./lstmp/trainLabel_kfold{}'.format(k_fold_num), train_sets.all_label)
        # Matrix_to_CSV('./data/res50/trainLabel_kfold{}'.format(feature_num__s, k_fold_num), train_sets.all_label)
        # test
        # Matrix_to_CSV('./data/res50/testLabel_kfold{}'.format(feature_num__s, k_fold_num), test_sets.all_label)
        Matrix_to_CSV('./lstmp/testLabel_kfold{}'.format(k_fold_num), test_sets.all_label)


    # sess.close()
    # print('All time:', time.time() - time1)

if __name__ == '__main__':
    main()
