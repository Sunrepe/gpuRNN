# 用于测试新数据是否和老数据可以共用
from data_pre.alldata import *
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tmp_trans_wavelet

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
learning_rate = 0.0025
lambda_loss_amount = 0.0055
training_iters = 150  # Loop 1000 times on the dataset
batch_size = 400
display_iter = 2000  # To show test set accuracy during training
model_save = 20

k_fold_num = 0
fold = './data/actdata/'
savename = '_selftest_kfold'+str(k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def LSTM_RNN(_X, seqlen, _weight, _bias):
    with tf.variable_scope('dwt0'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X[0], sequence_length=tf.to_int32(seqlen[0]), dtype=tf.float32)
        lstm_out0 = tf.divide(tf.reduce_sum(outputs, 1), seqlen[0][:, None])
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X[1], sequence_length=tf.to_int32(seqlen[1]), dtype=tf.float32)
        lstm_out1 = tf.divide(tf.reduce_sum(outputs, 1), seqlen[1][:, None])
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X[2], sequence_length=tf.to_int32(seqlen[2]), dtype=tf.float32)
        lstm_out2 = tf.divide(tf.reduce_sum(outputs, 1), seqlen[2][:, None])
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X[3], sequence_length=tf.to_int32(seqlen[3]), dtype=tf.float32)
        lstm_out3 = tf.divide(tf.reduce_sum(outputs, 1), seqlen[3][:, None])

    # print(type(lstm_out0.shape))
    # print(type(lstm_out1.shape))
    # print(type(lstm_out2.shape))
    # print(type(lstm_out3.shape))
    lstm_out0 = tf.concat([lstm_out0, lstm_out1], 0)
    lstm_out0 = tf.concat([lstm_out0, lstm_out2], 0)
    lstm_out0 = tf.concat([lstm_out0, lstm_out3], 0)
    with tf.variable_scope('fullConnect'):
        lstm_out = tf.layers.dense(lstm_out0, 50)

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']


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



def LSTM_RNN_static(_X, seqlen, _weight, _bias):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cell_3 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3])
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # Get LSTM cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X, sequence_length=seqlen, dtype=tf.float32)
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    # 获取数据,此时维度为[none,batch_size,n_hidden],需要进一步降维
    # lstm_out = tf.batch_gather(outputs, tf.to_int32(seqlen[:, None]-1))
    # lstm_out = tf.reshape(lstm_out, [-1, n_hidden])
    # 方案二：
    lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']


def LSTM_RNN_WT(_X, seqlen, _weight, _bias):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cell_3 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3])
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # Get LSTM cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X, sequence_length=seqlen, dtype=tf.float32)
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    # 获取数据,此时维度为[none,batch_size,n_hidden],需要进一步降维
    lstm_out = tf.batch_gather(outputs, tf.to_int32(seqlen[:, None]-1))
    lstm_out = tf.reshape(lstm_out, [-1, n_hidden])
    # 方案二：
    # lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']


def main():
    time1 = time.time()
    # tmp_trans_wavelet.main_datatrans(fold)
    print('loading data...')
    train_sets = All_data_merge(foldname=fold, max_seq=max_seq,
                                num_class=n_classes, trainable=True, kfold_num=k_fold_num)
    test_sets = All_data_merge(foldname=fold, max_seq=max_seq,
                               num_class=n_classes, trainable=False, kfold_num=k_fold_num)
    train_data_len = len(train_sets.all_label)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time()-time1)

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

    # Graph weights
    # weights = {
    #     'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    # }
    # biases = {
    #     'out': tf.Variable(tf.random_normal([n_classes]))
    # }
    #
    # weights = {
    #     'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),  # Hidden layer weights
    #     'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    # }
    # biases = {
    #     'hidden': tf.Variable(tf.random_normal([n_hidden])),
    #     'out': tf.Variable(tf.random_normal([n_classes]))
    # }

    preds = LSTM_RNN_tmp(x0,x1,x2,x3,x4,x5,x6,x7,x8,
                        seq_len0,seq_len1,seq_len2,seq_len3,
                        seq_len4, seq_len5, seq_len6, seq_len7,seq_len8)
    costs = []
    for i_preds in range(len(preds)):
        costs.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=preds[i_preds])))

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    # L2 loss prevents this overkill neural network to overfit the data

    cost = l2+sum(costs)  # Softmax loss
    pred = sum(preds)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=12)
    # start train and test
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    # saver.restore(sess, "./lstm/model_mergeall_kfold4.ckpt-200")
    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    print("Start train!")

    while step * batch_size <= training_iters * train_data_len:
        batch_ys, batch_xs, batch_seq_len = train_sets.next(batch_size)
        feed_dic = {
            y: batch_ys,
            x0: batch_xs[0],
            x1: batch_xs[1],
            x2: batch_xs[2],
            x3: batch_xs[3],
            x4: batch_xs[4],
            x5: batch_xs[5],
            x6: batch_xs[6],
            x7: batch_xs[7],
            x8: batch_xs[8],
            seq_len0: batch_seq_len[0],
            seq_len1: batch_seq_len[1],
            seq_len2: batch_seq_len[2],
            seq_len3: batch_seq_len[3],
            seq_len4: batch_seq_len[4],
            seq_len5: batch_seq_len[5],
            seq_len6: batch_seq_len[6],
            seq_len7: batch_seq_len[7],
            seq_len8: batch_seq_len[8]
        }
        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict=feed_dic
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (
                step * batch_size > training_iters * train_data_len):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))
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

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict=feed_dic
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        # save the model:
        if (step * batch_size % (display_iter * model_save) == 0) or (
                        step * batch_size > training_iters * train_data_len):
            save_path = saver.save(sess, "./lstm2/model{}.ckpt".format(savename), global_step=step)
            print("Model saved in file: %s" % save_path)
            # save loss and acc
            # save and load
            Matrix_to_CSV(
                './loss_dir2/{}_hd{}iter{}ba{}lr{}train_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                         learning_rate), train_losses)
            Matrix_to_CSV(
                './loss_dir2/{}_hd{}iter{}ba{}lr{}train_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                        learning_rate), train_accuracies)
            Matrix_to_CSV(
                './loss_dir2/{}_hd{}iter{}ba{}lr{}test_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                        learning_rate), test_losses)
            Matrix_to_CSV(
                './loss_dir2/{}_hd{}iter{}ba{}lr{}test_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                       learning_rate), test_accuracies)
        step += 1

    print("Optimization Finished!")

    # Accuracy for test data
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
    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict=feed_dic
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)
    # save and load
    Matrix_to_CSV(
        './loss_dir2/{}_hd{}iter{}ba{}lr{}train_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                 learning_rate), train_losses)
    Matrix_to_CSV(
        './loss_dir2/{}_hd{}iter{}ba{}lr{}train_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                learning_rate), train_accuracies)
    Matrix_to_CSV(
        './loss_dir2/{}_hd{}iter{}ba{}lr{}test_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                learning_rate), test_losses)
    Matrix_to_CSV('./loss_dir2/{}_hd{}iter{}ba{}lr{}test_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                         learning_rate), test_accuracies)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))
    print("All train time = {}".format(time.time() - time1))
    save_path = saver.save(sess, "./lstm2/model{}.ckpt-final".format(savename))
    print("Final Model saved in file: %s" % save_path)


    # plt.show()

    # train_losses = np.loadtxt('./loss_dir/train_loss.txt')
    # train_accuracies = np.loadtxt('../loss_dir/train_acc.txt')
    # test_losses = np.loadtxt('../loss_dir/test_loss.txt')
    # test_accuracies = np.loadtxt('../loss_dir/test_acc.txt')

    predictions = one_hot_predictions.argmax(1)
    result_labels = test_sets.all_label.argmax(1)
    print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.yticks(tick_marks, LABELS)
    plt.savefig('./loss_dir2/Matrix{}.png'.format(savename), dpi=600, bbox_inches='tight')


    font = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 18
    }
    matplotlib.rc('font', **font)
    # matplotlib.use('Agg')
    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1]),
        [training_iters * train_data_len]
    )
    plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')
    plt.savefig('./loss_dir2/accloss_{}.png'.format(savename), dpi=600, bbox_inches='tight')

    sess.close()


if __name__ == '__main__':
    main()
