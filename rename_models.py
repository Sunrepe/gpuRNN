# 用于测试新数据是否和老数据可以共用
import tensorflow as tf
import os
import time
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # just error no warning

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 8
max_seq = 800
tmp_use_len = [150, 150, 250, 450, 800, 800, 800, 800, 400]

# Training

k_fold_num = 0
feature_num__s = 8
fold = './data/actdata/'


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def get_model(fea_num, kfold_num):
    pass
    if fea_num == 0 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 1 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 2 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 3 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2400'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 4 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2400'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 5 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2600'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 6 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 7 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-2000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 8 and kfold_num == 0:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-final'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 0 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 1 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 2 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    elif fea_num == 3 and kfold_num == 1:
        model_path = 'E:/Research-bachelor/storeMODELs/2dif_feature/' \
                     '{}/allmerge_f{}_kfold{}/' \
                     'model_feature{}_kfold{}.ckpt-3000'.format(fea_num,fea_num,kfold_num,fea_num,kfold_num)
    return model_path


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

    return lstm_out


def LSTM_RNN_f2(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('std'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f3(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('wlc'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f4(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f5(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt2'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f6(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt3'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f7(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('dwt4'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def LSTM_RNN_f8(x, seq, _weight, _bias):
    # dwt
    with tf.variable_scope('fft'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_cells, inputs=x, sequence_length=tf.to_int32(seq), dtype=tf.float32)
        lstm_out = tf.divide(tf.reduce_sum(outputs, 1), seq[:, None])

    return lstm_out


def main():

    time1 = time.time()
    print('Rename Models...')

    # Graph weights
    with tf.variable_scope("weight"):
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
    pred = LSTM_RNN_f8(x, seq_len, weights, biases)

    with tf.name_scope('fullConnect'):
        lstm_out = tf.matmul(pred, weights['out']) + biases['out']

    saver = tf.train.Saver(max_to_keep=12)

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

    saver.restore(sess, get_model(fea_num=feature_num__s, kfold_num=k_fold_num))

    save_path = saver.save(sess, "./models/fea{}/model_kfold{}.ckpt".format(feature_num__s, k_fold_num))
    print("Model Resave at {}".format(save_path))

    sess.close()
    print('All time:', time.time() - time1)

if __name__ == '__main__':
    main()
