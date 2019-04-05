'''
本函数用于测试所有训练好的结果.
将训练结果测试到每个人.
以验证不同数据集的效果
'''
# 用于测试新数据是否和老数据可以共用
from data_pre.dataFortest import *
import tensorflow as tf
import os
import time
import csv
import pandas as pd
import numpy as np
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning

# All hyperparameters
model_name = 'E:/Research-bachelor/storeMODELs/RNNCNN_data_and_model2/lstm/modelRNNCNN_newdatadrop_.ckpt-final'
df_save_path = 'E:/Research-bachelor/storeMODELs/RNNCNN_data_and_model2/RNNCNN0402_everyone.csv'
foldname = './data/test3/'

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 10
max_seq = 800

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = 500  # Loop 200 times on the dataset
batch_size = 100
display_iter = 4000  # To show test set accuracy during training
model_save = 50
savename = 'RNNCNN_newdatadrop_'
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
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
        conv1 = tf.nn.dropout(h_pool1, 0.7)

    # 第二层卷积
    with tf.name_scope('conv2'):
        w_conv2 = weight_init([5,1,4,2], 'conv2_w')
        b_conv2 = bias_init([2], 'conv2_b')
        conv2 = tf.nn.conv2d(input=conv1, filter=w_conv2, strides=[1,2,1,1], padding='VALID')
        h_conv2 = tf.nn.relu(conv2+b_conv2)
        # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    _a = h_conv2.shape
    return tf.reshape(h_conv2, [-1, _a[1], 8])


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

    # load sess and model
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

    saver = tf.train.Saver()
    print('load model...')
    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    saver.restore(sess, model_name)
    # load accomplished
    print('Model are loaded!\t{}s'.format(time.time()-time1))
    df_data = []
    print("Start test!")
    person_list = getAllPeople(foldname)
    for person in person_list:
        print(person)
        test_data = Per_CNNData(foldname=foldname,
                               person_name=person,
                               max_seq=max_seq,
                               num_class=n_classes)
        # Accuracy for test data
        one_hot_predictions = sess.run(
            pred,
            feed_dict={
                x: test_data.all_data,
                y: test_data.all_label,
                seq_len: test_data.all_seq_len
            }
        )
        predictions = one_hot_predictions.argmax(1)
        result_labels = test_data.all_label.argmax(1)
        _percision = metrics.precision_score(result_labels, predictions, average="weighted")
        _recall = metrics.recall_score(result_labels, predictions, average="weighted")
        _f1Score = metrics.f1_score(result_labels, predictions, average="weighted")
        print("Precision: {}%".format(100 * _percision))
        print("Recall: {}%".format(100 * _recall))
        print("f1_score: {}%".format(100 * _f1Score))

        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        print(confusion_matrix)
        n_wrong=0
        for i in range(len(predictions)):
            if predictions[i]!=result_labels[i]:
                n_wrong+=1
                # print('True:',LABELS[result_labels[i]],'Pred:',LABELS[predictions[i]])
        print(person, "\t total wrong pred:{}".format(n_wrong))
        df_data.append([_recall, _percision, _f1Score, n_wrong])
        print("")

    df = pd.DataFrame(df_data, index=person_list, columns=["Recall", "Precision", "F1_score", "Total_wrong"])
    df.to_csv(df_save_path)

    sess.close()


if __name__ == '__main__':
    main()