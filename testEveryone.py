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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800
k_fold_num = 1
# model_name = 'E:/Research-bachelor/storeMODELs/5kfold/lstmkfold{}/lstm/model_LSTM_kfold{}.ckpt-final'.format(k_fold_num,k_fold_num)
df_save_path = 'E:/Research-bachelor/storeMODELs/5kfold/LSTMnewdata_everyone_kfold{}.csv'.format(k_fold_num)
model_name = "E:/Research-bachelor/storeMODELs/5kfold/lstm_kfold1/lstm/model_LSTM_kfold1.ckpt-final"
foldname = './data/actdata/'

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def getPersons(foldname, kfold_num):
    '''
        根据文件夹获得获得所有人,并根据kfold_num将所有人分类为训练集/测试集人物
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    _person = list(_person)
    _person.sort()
    _person.remove('marui')
    _person.remove('zhangyixuan')
    test_p = _person[7*kfold_num:7*(kfold_num+1)]
    # train_p = ['marui', 'zhangyixuan']
    # for i in _person:
    #     if i not in test_p:
    #         train_p.append(i)
    return test_p


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
    pred = LSTM_RNN(x, seq_len, weights, biases)
    # Loss, optimizer and evaluation
    saver = tf.train.Saver()
    # start train and test
    # To keep track of training's performance

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    assert os.path.exists(foldname)
    saver.restore(sess, model_name)
    # load accomplished
    print('Model are loaded!\t{}s'.format(time.time()-time1))

    df_data = []
    print("Start test!")
    person_list = getPersons(foldname, k_fold_num)
    print("person:",person_list)
    for person in person_list:
        print(person)
        test_data = Per_RNNData(foldname=foldname,
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
        # print("Precision: ",metrics.precision_score(result_labels, predictions, average=None))
        # # print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions)))
        # # print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions)))

        print("{}'s Confusion Matrix:".format(person))
        confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
        print(confusion_matrix)
        n_wrong = 0
        for i in range(len(predictions)):
            if predictions[i] != result_labels[i]:
                n_wrong += 1
                # print('True:', LABELS[result_labels[i]], 'Pred:', LABELS[predictions[i]])
        print(person, "\t total wrong pred:{}".format(n_wrong))
        print("")
        df_data.append([_recall, _percision, _f1Score, n_wrong])
    df=pd.DataFrame(df_data, index=person_list, columns=["Recall","Precision","F1_score","Total_wrong"])
    df.to_csv(df_save_path)
    sess.close()


if __name__ == '__main__':
    main()