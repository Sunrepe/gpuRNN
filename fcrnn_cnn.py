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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warnings and errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_concat = 450

# Training
learning_rate = 0.00005
lambda_loss_amount = 0.00015  # 目前最好参数 0.000025,0.00015
training_iters = 300  # Loop 1000 times on the dataset
batch_size = 800
display_iter = 8800  # To show test set accuracy during training
model_save = 80

k_fold_num = 4
fold = './data/res50/'

savename = './models/kfold{}/fcnet/fcnet_kfold{}'.format(k_fold_num, k_fold_num)

LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def Matrix_to_CSV_array(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def FC_Net(lstm_out, keep_pro):
    '''
    训练单独的FC网络
    :param lstm_out:
    :param keep_pro:
    :return:
    '''
    with tf.variable_scope('FC_Nets'):
        lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_pro)
        lstm_out = tf.layers.dense(lstm_out, 512)
        lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_pro)
        lstm_out = tf.layers.dense(lstm_out, 1024)
        lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_pro)
        lstm_out = tf.layers.dense(lstm_out, 512)
        lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_pro)
        lstm_out = tf.layers.dense(lstm_out, 256)
        lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_pro)
        lstm_out = tf.layers.dense(lstm_out, 128)
        lstm_out = tf.layers.dense(lstm_out, 10)

    return lstm_out


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = data_load_res(foldname=fold, trainable=True, kfold_num=k_fold_num)
    test_sets = data_load_res(foldname=fold, trainable=False, kfold_num=k_fold_num)
    train_data_len = len(train_sets.all_label)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time()-time1)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_concat])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    pred = FC_Net(x, keep_prob)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    # L2 loss prevents this overkill neural network to overfit the data

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=2)
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
    # saver.restore(sess, savename+'final')
    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    print("Start train!")

    while step * batch_size <= training_iters * train_data_len:
        batch_ys, batch_xs = train_sets.next(batch_size)

        if step % 1000 == 0:
            feed_dic = {
                y: test_sets.all_label,
                x: test_sets.data_res,
                keep_prob: 1.0
            }
        else:
            feed_dic = {
                y: batch_ys,
                x: batch_xs,
                keep_prob: 1.0
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
            print("Training step #" + str(step) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))
            feed_dic = {
                y: test_sets.all_label,
                x: test_sets.data_res,
                keep_prob: 1.0
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
        if (step * batch_size % (display_iter * 100) == 0) or (
                        step * batch_size > training_iters * train_data_len):
            save_path = saver.save(sess, savename, global_step=step)
            print("Model saved in file: %s" % save_path)
        step += 1

    print("Optimization Finished!")

    # Accuracy for test data
    feed_dic = {
        y: test_sets.all_label,
        x: test_sets.data_res,
        keep_prob: 1.0
    }
    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict=feed_dic
    )

    Matrix_to_CSV_array('./data/res10/test/fea9_kfold{}'.format(k_fold_num), one_hot_predictions)
    # Matrix_to_CSV_array('./data/res10/label_fea9_kfold{}'.format(k_fold_num), test_sets.all_label)

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))
    print("All train time = {}".format(time.time() - time1))
    save_path = saver.save(sess, savename+'-final')
    print("Final Model saved in file: %s" % save_path)
    #
    # font = {
    #     'family': 'Times New Roman',
    #     'weight': 'bold',
    #     'size': 18
    # }
    # matplotlib.rc('font', **font)
    #
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
    #     [training_iters * train_data_len]
    # )
    # plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
    # plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
    #
    # plt.title("Training session's progress over iterations")
    # plt.legend(loc='upper right', shadow=True)
    # plt.ylabel('Training Progress (Loss or Accuracy values)')
    # plt.xlabel('Training iteration')
    # plt.savefig('./loss_dir/accloss.png', dpi=600, bbox_inches='tight')
    #
    # # plt.show()
    #
    # # save and load
    # Matrix_to_CSV(
    #     './loss_dir/{}_hd{}iter{}ba{}lr{}train_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
    #                                                             learning_rate), train_losses)
    # Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}train_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
    #                                                                      learning_rate), train_accuracies)
    # Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}test_loss.txt'.format(savename, n_hidden, training_iters, batch_size,
    #                                                                      learning_rate), test_losses)
    # Matrix_to_CSV('./loss_dir/{}_hd{}iter{}ba{}lr{}test_acc.txt'.format(savename, n_hidden, training_iters, batch_size,
    #                                                                     learning_rate), test_accuracies)
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
    Matrix_to_CSV_array(filename='./matrix/all_lstm/2-matrix_kfold{}.txt'.format(k_fold_num), data=confusion_matrix)
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
    plt.savefig('./loss_dir/2-Matrix_kfold{}.png'.format(k_fold_num), dpi=300, bbox_inches='tight')

    sess.close()


if __name__ == '__main__':
    main()