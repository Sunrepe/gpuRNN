from data_pre.alldata import AllData
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# LSTM Neural Network's internal structure

n_hidden = 32  # Hidden layer num of features
n_classes = 8  # Total classes (should go up, or should go down)
max_seq = 300
n_inputs = 8

# Training
learning_rate = 0.025
lambda_loss_amount = 0.015
training_iters = 200  # Loop 1000 times on the dataset
batch_size = 80
display_iter = 2400  # To show test set accuracy during training
save_num = 40


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def BiLSTM_RNN(_X, seqlen, _weight, _bias,):
    # forward
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
    lstm_out_1 = tf.divide(tf.reduce_sum(_out1, 1), seqlen[:, None])
    lstm_out_2 = tf.divide(tf.reduce_sum(_out2, 1), seqlen[:, None])
    return tf.matmul(lstm_out_1 + lstm_out_2, _weight['out']) + _bias['out']
    # return outputs
    # many to one 关键。两种方案，一个是选择最后的输出，一个是选择所有输出的均值
    # 方案一：
    # lstm_out = tf.gather_nd(outputs, seqlen-1)
    # 方案二：
    # lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seqlen[:, None])


def main():
    time1 = time.time()
    train_sets = AllData(foldname='../data/train/')
    test_sets = AllData(foldname='../data/test/')
    train_data_len = len(train_sets.all_seq_len)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])

    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = BiLSTM_RNN(x, seq_len, weights, biases)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    # start train and test
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
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
                seq_len: batch_seq_len
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size >= training_iters*train_data_len):
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
                    seq_len: test_sets.all_seq_len
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        # save the model:
        if (step * batch_size % (display_iter*save_num) == 0) or (
                step * batch_size >= training_iters * train_data_len):
            save_path = saver.save(sess, "../model/bilstm_model/model.ckpt", global_step=step)
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
    print("All train time = {}".format(time.time()-time1))

    # save and load
    Matrix_to_CSV('../loss_dir/bi_lstm/train_loss_1.txt', train_losses)
    Matrix_to_CSV('../loss_dir/bi_lstm/train_acc_1.txt', train_accuracies)
    Matrix_to_CSV('../loss_dir/bi_lstm/test_loss_1.txt', test_losses)
    Matrix_to_CSV('../loss_dir/bi_lstm/test_acc_1.txt', test_accuracies)
    # train_losses = np.loadtxt('./loss_dir/train_loss.txt')
    # train_accuracies = np.loadtxt('./loss_dir/train_acc.txt')
    # test_losses = np.loadtxt('./loss_dir/test_loss.txt')
    # test_accuracies = np.loadtxt('./loss_dir/test_acc.txt')

    # matrix
    # predictions = one_hot_predictions.argmax(1)
    # y_test = test_sets.all_label.argmax(1)
    #
    # print("Precision: {}%".format(100 * metrics.precision_score(y_test, predictions, average="weighted")))
    # print("Recall: {}%".format(100 * metrics.recall_score(y_test, predictions, average="weighted")))
    # print("f1_score: {}%".format(100 * metrics.f1_score(y_test, predictions, average="weighted")))
    #
    # print("")
    # print("Confusion Matrix:")
    # confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    # print(confusion_matrix)
    # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

    # print("Confusion matrix (normalised to % of total test data):")
    # print(normalised_confusion_matrix)
    # print("Note: training and testing data is not equally distributed amongst classes, ")
    # print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # font = {
    #     'family': 'Bitstream Vera Sans',
    #     'weight': 'bold',
    #     'size': 18
    # }
    # matplotlib.rc('font', **font)
    #
    # import numpy as np
    # width = 12
    # height = 12
    # plt.figure(figsize=(width, height))
    #
    # indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
    # plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
    # plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")
    #
    # indep_test_axis = np.append(
    #     np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1]),
    #     [training_iters]
    # )
    # plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
    # plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
    #
    #
    #
    # plt.title("Training session's progress over iterations")
    # plt.legend(loc='upper right', shadow=True)
    # plt.ylabel('Training Progress (Loss or Accuracy values)')
    # plt.xlabel('Training iteration')
    #
    # plt.show()

    sess.close()


if __name__ == '__main__':
    main()
