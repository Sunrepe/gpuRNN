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
model_name = 'E:/Research-bachelor/storeMODELs/BiLSTM0402_data_and_model/lstm/model_BiLSTM0402_.ckpt-final'
df_save_path = 'E:/Research-bachelor/storeMODELs/BiLSTM0402_data_and_model/BiLSTM0402_everyone.csv'
foldname = './data/test3/'

n_hidden = 40  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 700

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = 200  # Loop 1000 times on the dataset
batch_size = 100
display_iter = 3200  # To show test set accuracy during training
model_save = 20
savename = '_BiLSTM0402_'
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow([row])


def BiLSTM_RNN(_X, seqlen, _weight, _bias,):
    # shaping the dataSet
    # _X = tf.reshape(_X, [-1, n_inputs])
    # _X = tf.nn.relu(tf.matmul(_X, _weight['hidden']) + _bias['hidden'])
    # _X = tf.reshape(_X, [-1, max_seq, n_inputs])

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
    lstm_out_1 = tf.divide(tf.reduce_sum(_out1, 1), seqlen[:, None])
    lstm_out_2 = tf.divide(tf.reduce_sum(_out2, 1), seqlen[:, None])
    _out_last = lstm_out_1*0.7 + lstm_out_2*0.3
    return tf.matmul(_out_last, _weight['out']) + _bias['out']

def main():
    time1 = time.time()

    # load sess and model
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])
    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = BiLSTM_RNN(x, seq_len, weights, biases)
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