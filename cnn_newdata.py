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
n_inputs = 8
max_seq = 800

# Training
learning_rate = 0.0015
lambda_loss_amount = 0.0075
training_iters = 500  # Loop 200 times on the dataset
batch_size = 400
display_iter = 4000  # To show test set accuracy during training
model_save = 100

res_fold = './data/actdata/'
k_fold_num = 0
savename = 'emgandwavelet_CNN_kfold'+str(k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
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


def CNNnet2(inputs,keep_pro):
    '''
    CNN网络,用于获得动态长度的数据,之后交给RNN网络
    :param inputs:
    :return:
    '''
    # 第一层卷积

    with tf.name_scope('conv1'):
        w_conv1 = weight_init([5, 1, 1, 4], 'conv1_w')
        b_conv1 = bias_init([4], 'conv1_b')
        conv1 = tf.nn.conv2d(input=inputs, filter=w_conv1, strides=[1, 2, 1, 1], padding='VALID')
        conv1 = tf.nn.relu(conv1+b_conv1)
        # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],padding='VALID')
        # conv1 = tf.nn.dropout(conv1, keep_pro)

    # 第二层卷积
    with tf.name_scope('conv2'):
        w_conv2 = weight_init([5, 1, 4, 4], 'conv2_w')
        b_conv2 = bias_init([4], 'conv2_b')
        conv2 = tf.nn.conv2d(input=conv1, filter=w_conv2, strides=[1,2,1,1], padding='VALID')
        conv2 = tf.nn.relu(conv2+b_conv2)
        # conv2 = tf.nn.dropout(conv2,keep_pro)
        # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    with tf.name_scope('conv3'):
        w_conv3 = weight_init([10, 1, 4, 4], 'conv3_w')
        b_conv3 = bias_init([4], 'conv3_b')
        conv3 = tf.nn.conv2d(input=conv2, filter=w_conv3, strides=[1,2,1,1], padding='VALID')
        conv3 = tf.nn.relu(conv3+b_conv3)

    with tf.name_scope('conv4'):
        w_conv4 = weight_init([5, 1, 4, 2], 'conv4_w')
        b_conv4 = bias_init([2], 'conv4_b')
        conv4 = tf.nn.conv2d(input=conv3, filter=w_conv4, strides=[1,1,1,1], padding='VALID')
        # conv4 = tf.nn.relu()
        conv4 = conv4+b_conv4
    _a = conv4.shape
    return tf.reshape(conv4, [-1, _a[1], 16])


def LSTM_RNN(_X, seqlen, _weight, _bias):
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
    # Get LSTM cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=_X, sequence_length=seqlen, dtype=tf.float32)
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    #  获取数据,此时维度为[none,batch_size,n_hidden],需要进一步降维
    # lstm_out = tf.batch_gather(outputs, tf.to_int32(seqlen[:, None]-2))
    # lstm_out = tf.reshape(lstm_out, [-1, n_hidden])
    # 方案二：
    lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])

    return tf.matmul(lstm_out, _weight['out']) + _bias['out']


def BiLSTM_RNN(_X, seqlen, keep_pro):
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
    # tf.reshape(tf.batch_gather(outputs, tf.to_int32(seqlen[:, None] - 1)), [-1, n_hidden])
    #
    # lstm_out_1 = tf.reshape(tf.batch_gather(_out1, tf.to_int32(seqlen[:, None] - 2)), [-1, n_hidden])
    # lstm_out_2 = tf.reshape(tf.batch_gather(_out2, tf.to_int32(seqlen[:, None] - 2)), [-1, n_hidden])
    # 方案二
    lstm_out_1 = tf.divide(tf.reduce_sum(_out1, 1), seqlen[:, None])
    lstm_out_2 = tf.divide(tf.reduce_sum(_out2, 1), seqlen[:, None])
    _out_last = tf.concat([lstm_out_1, lstm_out_2], 1)
    # _out_last = tf.nn.dropout(_out_last, keep_prob=keep_pro)
    _out_last = tf.layers.dense(_out_last, 10)

    return _out_last


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = CNNData2(foldname=res_fold, max_seq=max_seq,
                             num_class=n_classes, trainable=True, kfold_num=k_fold_num)
    test_sets = CNNData2(foldname=res_fold, max_seq=max_seq,
                            num_class=n_classes, trainable=False, kfold_num=k_fold_num)
    train_data_len = len(train_sets.all_seq_len)
    print('train:', len(train_sets.all_seq_len), 'test:', len(test_sets.all_seq_len))
    print('load data time:', time.time()-time1)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32)


    CNN_res = CNNnet2(x, keep_prob)
    pred = BiLSTM_RNN(CNN_res, seq_len, keep_prob)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2  # Softmax loss
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

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    print("Start train!")

    while step * batch_size <= training_iters * train_data_len:
        batch_xs, batch_ys, batch_seq_len = train_sets.next(batch_size)
        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys,
                seq_len: batch_seq_len,
                keep_prob: 0.7
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % (display_iter*20) == 0) or (step == 1) or (
                step * batch_size > training_iters * train_data_len):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: test_sets.all_data,
                    y: test_sets.all_label,
                    seq_len: test_sets.all_seq_len,
                    keep_prob: 1.0
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        # save the model:
        if (step * batch_size % (display_iter * model_save) == 0) or (
                        step * batch_size > training_iters * train_data_len):
            save_path = saver.save(sess, "./lstm/model{}.ckpt".format(savename), global_step=step)
            print("Model saved in file: %s" % save_path)
            Matrix_to_CSV(
                './loss_dir/{}_hd{}iter{}ba{}lr{}train_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                        learning_rate), train_losses)
            Matrix_to_CSV(
                './loss_dir/{}_hd{}iter{}ba{}lr{}train_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                       learning_rate), train_accuracies)
            Matrix_to_CSV(
                './loss_dir/{}_hd{}iter{}ba{}lr{}test_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                       learning_rate), test_losses)
            Matrix_to_CSV(
                './loss_dir/{}_hd{}iter{}ba{}lr{}test_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                      learning_rate), test_accuracies)
        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: test_sets.all_data,
            y: test_sets.all_label,
            seq_len: test_sets.all_seq_len,
            keep_prob: 1.0
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))
    print("All train time = {}".format(time.time() - time1))
    save_path = saver.save(sess, "./lstm/model{}.ckpt-final".format(savename))
    print("Final Model saved in file: %s" % save_path)

    font = {
        'family': 'Bitstream Vera Sans',
        'weight': 'bold',
        'size': 18
    }
    matplotlib.rc('font', **font)

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
    plt.savefig('./lstm/accloss_{}.png'.format(savename), dpi=600, bbox_inches='tight')

    # plt.show()

    # save and load
    Matrix_to_CSV(
        './loss_dir/{}_hd{}iter{}ba{}lr{}train_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                learning_rate), train_losses)
    Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}train_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                         learning_rate), train_accuracies)
    Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}test_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                         learning_rate), test_losses)
    Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}test_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
                                                                        learning_rate), test_accuracies)
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
    plt.savefig('./lstm/Matrix{}.png'.format(savename), dpi=600, bbox_inches='tight')

    sess.close()


if __name__ == '__main__':
    main()