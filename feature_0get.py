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
lambda_loss_amount = 0.0015
training_iters = 150  # Loop 1000 times on the dataset
batch_size = 400
display_iter = 4000  # To show test set accuracy during training
model_save = 20

k_fold_num = 1
feature_num__s = 1
fold = './data/actdata/'
savename = '_feature{}_kfold{}'.format(feature_num__s, k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def LSTM_RNN_f0(x, seq):
    # dwt
    with tf.variable_scope('ori'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f1(x, seq):
    # dwt
    with tf.variable_scope('avg'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f2(x, seq):
    # dwt
    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f3(x, seq):
    # dwt
    with tf.variable_scope('wlc'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f4(x, seq):
    # dwt
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f5(x, seq):
    # dwt
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f6(x, seq):
    # dwt
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f7(x, seq):
    # dwt
    with tf.variable_scope('dwt4'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def LSTM_RNN_f8(x, seq):
    # dwt
    with tf.variable_scope('fft'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.variable_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.layers.dense(lstm_out, 10)
    return lstm_out


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = All_data_feature(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=True, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    test_sets = All_data_feature(foldname=fold, max_seq=max_seq,
                     num_class=10, trainable=False, kfold_num=k_fold_num,
                     feature_num=feature_num__s)
    train_data_len = len(train_sets.all_seq_len)
    print('train:', len(train_sets.all_seq_len), 'test:', len(test_sets.all_seq_len))
    print('load data time:', time.time() - time1)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])

    pred = LSTM_RNN_f1(x, seq_len)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    # L2 loss prevents this overkill neural network to overfit the data

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2  # Softmax loss

    learning_rate = tf.Variable(0.0025, dtype=tf.float32, trainable=False)
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
    # init = tf.global_variables_initializer()
    # sess.run(init)
    saver.restore(sess, './lstm_model/model_feature1_kfold0.ckpt-3200')
    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    print("Start train!")

    while step * batch_size <= training_iters * train_data_len:
        # 调整lr
        if step < 800:
            t = sess.run(tf.assign(learning_rate, 0.001))
        elif step < 1600:
            t = sess.run(tf.assign(learning_rate, 0.0005))
        else:
            t = sess.run(tf.assign(learning_rate, 0.00025))

        # learning_rate = cal_lr(learning_rate, step)
        batch_xs, batch_ys, batch_seq_len = train_sets.next(batch_size)
        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys,
                seq_len: batch_seq_len
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (
                        step * batch_size > training_iters * train_data_len):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ",  Accuracy = {:.6f}".format(acc) + \
                  ",  lr = {:.6f}".format(t))

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: test_sets.all_data,
                    y: test_sets.all_label,
                    seq_len: test_sets.all_seq_len
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        # save the model:
        if (step * batch_size % (display_iter * 20) == 0) or (
                        step * batch_size > training_iters * train_data_len):
            save_path = saver.save(sess, "./lstm/model{}.ckpt".format(savename), global_step=step)
            print("Model saved in file: %s" % save_path)
        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: test_sets.all_data,
            y: test_sets.all_label,
            seq_len: test_sets.all_seq_len
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
        'family': 'Times New Roman',
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
    plt.savefig('./loss_dir/accloss_{}.png'.format(savename), dpi=600, bbox_inches='tight')

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
    plt.savefig('./loss_dir/Matrix{}.png'.format(savename), dpi=600, bbox_inches='tight')

    sess.close()


if __name__ == '__main__':
    main()
