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


def f1scores(maxix,rangese):
    recdi = np.sum(maxix,axis=1)
    predi = np.sum(maxix,axis=0)
    print(recdi)
    # print(predi)
    rec = []
    pre = []
    f1sc = []
    for i in range(rangese):
        tmp1 = maxix[i,i]/recdi[i]
        tmp2 = maxix[i,i]/predi[i]
        f1sc.append(2.0/(1.0/tmp1+1.0/tmp2)*10000)
        rec.append(tmp1*10000)
        pre.append(tmp2*10000)
    accu = sum(rec)/rangese
    return [['Recall:',np.array(rec,dtype='int')],['Precision:',np.array(pre,dtype='int')],['F1score:',np.array(f1sc,dtype='int')],accu]


def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def Read_data_res(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 组合的数据
    '''
    my_matrix = np.loadtxt(filename, dtype='float', delimiter=",")
    return my_matrix


def get_one_hot_pre(n_fea):
    file = './data/res10/test/fea{}_kfold0'.format(n_fea)
    res = Read_data_res(file)
    for i_kfold in range(1, 5):
        file = './data/res10/test/fea{}_kfold{}'.format(n_fea, i_kfold)
        _tmp = Read_data_res(file)
        res = np.concatenate([res, _tmp], 0)
    return res


def get_one_hot_pre_kfold(n_fea, n_kfold):
    file = './data/res10/test/fea{}_kfold{}'.format(n_fea, n_kfold)
    res = Read_data_res(file)
    return res


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


def main5():
    '''
    计算不同类输出结果，并作简要分析。
    具体方法：根据不同类进行结果分析。
    :return:
    '''
    # LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    stream = ["原始信号", "均值", "标准差", "波长变化", "dwt1", "dwt2", "dwt3", "dwt4", "fft", "融合"]
    print("------------------------------")
    # label_one_hot = Read_data_res('./data/res10/all/label')
    # fea_num = 0
    i_fea = 4
    i_kfold = 1
    print(stream[i_fea], "Kfold{}".format(i_kfold))
    one_hot_pre = get_one_hot_pre_kfold(i_fea, i_kfold)
    label = Read_data_res('./data/res10/testLabel_kfold{}'.format(i_kfold))
    predictions = one_hot_pre.argmax(1)
    result_labels = label.argmax(1)

    print(
        "Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    # Matrix_to_CSV(filename='./matrix/all1_feas/matrix_fea{}.txt'.format(fea_num), data=confusion_matrix)
    print(confusion_matrix)


if __name__ == '__main__':
    main5()
