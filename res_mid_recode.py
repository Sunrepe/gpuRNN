# 用于测试新数据是否和老数据可以共用
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
    '''
    根据训练好的45个model
    获得res10、 res50
    :return:
    '''
    # time0 = time.time()

    time1 = time.time()
    # Graph weights
    with tf.variable_scope("weight"):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
    #
    # # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])
    #
    # pred = LSTM_RNN_f1(x, seq_len)
    k_fold_num = 0
    feature_num__s = 7
    savename = '_feature{}_kfold{}'.format(feature_num__s, k_fold_num)
    pred = LSTM_RNN_f7(x, seq_len, weights, biases)
    #
    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(pred, weights['out']) + biases['out']

    correct_pred = tf.equal(tf.argmax(lstm_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver(max_to_keep=12)

    print('loading data...')
    train_sets = All_data_feature_test(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=True, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    test_sets = All_data_feature_test(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=False, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time() - time1)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    saver.restore(sess, "./models/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s, k_fold_num))

    # save 中间结果
    res50, res10 = sess.run(
        [pred, lstm_out],
        feed_dict={
            x: train_sets.all_data,
            y: train_sets.all_label,
            seq_len: train_sets.all_seq_len
        }
    )
    Matrix_to_CSV('./data/res50/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    Matrix_to_CSV('./data/res10/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)
    # Matrix_to_CSV('./data/res50/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), train_sets.all_label)
    # Matrix_to_CSV('./data/res10/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), train_sets.all_label)

    # one_hot_predictions, accuracy = sess.run(
    #     [lstm_out, accuracy],
    #     feed_dict={
    #         x: test_sets.all_data,
    #         y: test_sets.all_label,
    #         seq_len: test_sets.all_seq_len
    #     }
    # )
    res50, res10 = sess.run(
        [pred, lstm_out],
        feed_dict={
            x: test_sets.all_data,
            y: test_sets.all_label,
            seq_len: test_sets.all_seq_len
        }
    )
    Matrix_to_CSV('./data/res50/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    Matrix_to_CSV('./data/res10/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)
    if feature_num__s == 0:
        # train
        Matrix_to_CSV('./data/res10/trainLabel_kfold{}'.format(feature_num__s, k_fold_num), train_sets.all_label)
        Matrix_to_CSV('./data/res50/trainLabel_kfold{}'.format(feature_num__s, k_fold_num), train_sets.all_label)
        # test
        Matrix_to_CSV('./data/res50/testLabel_kfold{}'.format(feature_num__s, k_fold_num), test_sets.all_label)
        Matrix_to_CSV('./data/res10/testLabel_kfold{}'.format(feature_num__s, k_fold_num), test_sets.all_label)
    # Matrix_to_CSV('./data/res50/test/label_fea{}_kfold{}'.format(feature_num__s, k_fold_num), test_sets.all_label)
    # Matrix_to_CSV('./data/res10/test/label_fea{}_kfold{}'.format(feature_num__s, k_fold_num), test_sets.all_label)

    # predictions = one_hot_predictions.argmax(1)
    # result_labels = test_sets.all_label.argmax(1)
    # print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    # print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    # print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    # print("")
    # print("Confusion Matrix:")
    # confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    # print(confusion_matrix)

    # sess.close()
    print('All time:', time.time() - time1)


def main2():
    '''
    在GCP-instance 中进行res10 、res50收集
    使用了所有训练好的model。（一共45个model）
    :return:
    '''
    # time0 = time.time()

    time1 = time.time()
    # Graph weights
    with tf.variable_scope("weight"):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
    #
    # # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])
    #
    # pred = LSTM_RNN_f1(x, seq_len)

    k_fold_num = 1
    feature_num__s = 8
    pred = LSTM_RNN_f8(x, seq_len, weights, biases)

    # savename = '_feature{}_kfold{}'.format(feature_num__s, k_fold_num)
    #
    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(pred, weights['out']) + biases['out']

    correct_pred = tf.equal(tf.argmax(lstm_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver(max_to_keep=12)
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    saver.restore(sess, "./models/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s, k_fold_num))


    print('loading data...')
    print("Fea:{}   Kfold:{}".format(feature_num__s, k_fold_num))
    train_sets = All_data_feature_test(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=True, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    test_sets = All_data_feature_test(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=False, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time() - time1)

# ----------------------
    # save 中间结果
    # train
    res50, res10, acc = sess.run(
        [pred, lstm_out, accuracy],
        feed_dict={
            x: train_sets.all_data,
            y: train_sets.all_label,
            seq_len: train_sets.all_seq_len
        }
    )
    Matrix_to_CSV('./datas/res50/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    Matrix_to_CSV('./datas/res10/train/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)
    # test
    res50, res10, acc = sess.run(
        [pred, lstm_out, accuracy],
        feed_dict={
            x: test_sets.all_data,
            y: test_sets.all_label,
            seq_len: test_sets.all_seq_len
        }
    )
    Matrix_to_CSV('./datas/res50/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res50)
    Matrix_to_CSV('./datas/res10/test/fea{}_kfold{}'.format(feature_num__s, k_fold_num), res10)

    # labels ----------
    Matrix_to_CSV('./datas/res10/trainLabel_kfold{}'.format(k_fold_num), train_sets.all_label)
    Matrix_to_CSV('./datas/res50/trainLabel_kfold{}'.format(k_fold_num), train_sets.all_label)
    Matrix_to_CSV('./datas/res50/testLabel_kfold{}'.format(k_fold_num), test_sets.all_label)
    Matrix_to_CSV('./datas/res10/testLabel_kfold{}'.format(k_fold_num), test_sets.all_label)
    # labels ----------

    predictions = res10.argmax(1)
    result_labels = test_sets.all_label.argmax(1)
    print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    print(confusion_matrix)
    print("Accuracy:{}".format(acc))
    # sess.close()
    print('All time:', time.time() - time1)
# --------------------
    sess.close()


def main3():
    '''
    测试如何更改参数，而不用重新训练。
    :return:
    '''
    # time0 = time.time()
    wei_name = ['yuan', 'pin', 'biao', 'bobian', 'zhou', 'xu', 'feng', 'yan', 'xin']
    lstm_name = ['ori', 'avg', 'std', 'wlc', 'dwt1', 'dwt2', 'dwt3', 'dwt4', 'fft']
    time1 = time.time()

    with tf.variable_scope("weight"):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    for i_fea in range(8, 9):
        g = tf.Graph()
        with tf.variable_scope(wei_name[i_fea]):
            weights0 = tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
            biases0 = tf.Variable(tf.random_normal([n_classes]))

        # Graph input/output
        x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])
        seq_len = tf.placeholder(tf.float32, [None])

        feature_num__s = i_fea
        if feature_num__s == 0:
            pred = LSTM_RNN_f0(x, seq_len, weights, biases)
        elif feature_num__s == 1:
            pred = LSTM_RNN_f1(x, seq_len, weights, biases)
        elif feature_num__s == 2:
            pred = LSTM_RNN_f2(x, seq_len, weights, biases)
        elif feature_num__s == 3:
            pred = LSTM_RNN_f3(x, seq_len, weights, biases)
        elif feature_num__s == 4:
            pred = LSTM_RNN_f4(x, seq_len, weights, biases)
        elif feature_num__s == 5:
            pred = LSTM_RNN_f5(x, seq_len, weights, biases)
        elif feature_num__s == 6:
            pred = LSTM_RNN_f6(x, seq_len, weights, biases)
        elif feature_num__s == 7:
            pred = LSTM_RNN_f7(x, seq_len, weights, biases)
        elif feature_num__s == 8:
            pred = LSTM_RNN_f8(x, seq_len, weights, biases)
        with tf.name_scope('fullConnect'):
            lstm_out = tf.matmul(pred, weights['out']) + biases['out']

        for i_kfold in range(5):
            k_fold_num = i_kfold
            saver = tf.train.Saver(max_to_keep=12)
            var = tf.global_variables()
            # for i in var:
            #     print(i)
            var_flow_restore00 = [val for val in var if lstm_name[i_fea] in val.name]
            print('var_flow_restore00:')
            for i in var_flow_restore00:
                print(i)
            var_flow_restore01 = [val for val in var if 'weight' in val.name]  # 取出名字中有‘flownet’的参数
            print('var_flow_restore01:')
            for i in var_flow_restore01:
                print(i)

            saver00 = tf.train.Saver(var_flow_restore00)  # 这句话就是关键了，可以网Saver中传参数
            saver01 = tf.train.Saver(var_flow_restore01)

            sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
            # sess = tf.InteractiveSession(g)
            saver00.restore(sess,
                            "./models/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s, k_fold_num))
            saver01.restore(sess,
                            "./models/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s, k_fold_num))

            w = sess.run(tf.assign(weights0, weights['out']))
            b = sess.run(tf.assign(biases0, biases['out']))
            # a = sess.run(var)
            # print(w)
            # print(b)
            save_path = saver.save(sess,
                                   "./modelsnew/kfold{}/fea{}/model_kfold{}.ckpt".format(k_fold_num, feature_num__s,
                                                                                         k_fold_num))
            print("Model saved in file: %s" % save_path)

            sess.close()
    print("All time:{}".format(time.time() - time1))

if __name__ == '__main__':
    main3()
