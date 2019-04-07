from data_pre.alldata import *
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 10
max_seq = 800

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = 200  # Loop 200 times on the dataset
batch_size = 100
display_iter = 1000  # To show test set accuracy during training
model_save = 100

k_fold_num = 0
savename = '_CNN_kfold'+str(k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def weight_init(shape, name):
    '''
    获取某个shape大小的参数
    '''
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05))


def bias_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def CNNnet(inputs):
    '''
    CNN网络,用于获得动态长度的数据,之后交给RNN网络
    :param inputs:
    :return:
    '''

    # 第一层卷积
    with tf.name_scope('conv1'):
        w_conv1 = weight_init([5, 3, 1, 4], 'conv1_w')
        b_conv1 = bias_init([4], 'conv1_b')
        conv1 = tf.nn.conv2d(input=inputs, filter=w_conv1, strides=[1,2,1,1], padding='VALID')
        h_conv1 = tf.nn.relu(conv1+b_conv1)
        # h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
        conv1 = tf.nn.dropout(h_conv1, 0.7)

    # 第二层卷积
    with tf.name_scope('conv2'):
        w_conv2 = weight_init([10, 1, 4, 2], 'conv2_w')
        b_conv2 = bias_init([2], 'conv2_b')
        conv2 = tf.nn.conv2d(input=conv1, filter=w_conv2, strides=[1,2,1,1], padding='VALID')
        h_conv2 = tf.nn.relu(conv2+b_conv2)
        # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    _a = h_conv2.shape
    return tf.reshape(h_conv2, [-1, _a[1], 16])


def LSTM_RNN(_X, seqlen, _weight, _bias):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # Get LSTM cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X, sequence_length=seqlen, dtype=tf.float32)
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    # lstm_out = tf.gather_nd(outputs, seqlen-1)
    # 方案二：
    lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = CNNData(foldname='./data/actdata/', max_seq=max_seq,
                             num_class=n_classes, trainable=True, kfold_num=k_fold_num)
    test_sets = CNNData(foldname='./data/actdata/', max_seq=max_seq,
                            num_class=n_classes, trainable=False, kfold_num=k_fold_num)
    train_data_len = len(train_sets.all_seq_len)
    print('load data time:',time.time()-time1)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])

    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    CNN_res = CNNnet(x)
    pred = LSTM_RNN(CNN_res, seq_len, weights, biases)


    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.close()


if __name__ == '__main__':
    main()