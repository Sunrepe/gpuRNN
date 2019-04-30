# 用于测试新数据是否和老数据可以共用
import tensorflow as tf
import os
import time
import csv

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

k_fold_num = 0
feature_num__s = 0
fold = './data/actdata/'
savename = '_feature{}_kfold{}'.format(feature_num__s, k_fold_num)
LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow([row])


def get_model(fea_num, kfold_num):
    pass
    if fea_num == 0 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/0/allmerge_f0_kfold0/model_feature0_kfold0.ckpt-3000'


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

    with tf.name_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f2(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.name_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f3(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('wlc'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.name_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f4(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f5(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f6(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.name_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f7(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt4'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])
    with tf.name_scope('fullConnect'):
        # lstm_out = tf.nn.dropout(lstm_out, keep_prob=0.8)
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def LSTM_RNN_f8(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('fft'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(lstm_out, _weight['out']) + _bias['out']
    return lstm_out


def main():
    time1 = time.time()
    print('loading model...')

    # Graph weights
    with tf.variable_scope("weight"):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    with tf.variable_scope("weight0"):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    seq_len = tf.placeholder(tf.float32, [None])

    # pred = LSTM_RNN_f1(x, seq_len)
    pred = LSTM_RNN_f0(x, seq_len, weights, biases)
    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(pred, weights['out']) + biases['out']

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

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    # init = tf.global_variables_initializer()
    # sess.run(init)
    saver.restore(sess,
                  'E:/Research-bachelor/storeMODELs/2dif_feature/0/allmerge_f0_kfold0/model_feature0_kfold0.ckpt-3000')

    print('load data time:', time.time() - time1)
    sess.close()


if __name__ == '__main__':
    main()
