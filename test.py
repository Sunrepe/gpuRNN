import os
import csv
import time
import shutil
import numpy as np
import pywt
import tensorflow as tf
import tmp_trans_wavelet

batch_size = 2
max_seq = 700
n_classes = 10
n_inputs = 8
n_hidden = 4


def main():
    x = tf.placeholder(tf.float32, [None, max_seq, n_inputs])
    x2 = tf.placeholder(tf.float32, [None, max_seq, n_inputs])

    with tf.variable_scope('rnn_1'):
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(10, forget_bias=1.0, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell_1, inputs=x, dtype=tf.float32)
        lstm_out_1 = outputs[-10]
    with tf.variable_scope('rnn_2'):
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(10, forget_bias=1.0, state_is_tuple=True)
        outputs2, _2 = tf.nn.dynamic_rnn(lstm_cell_2, inputs=x2, dtype=tf.float32)
        lstm_out_2 = outputs2[-10]

    lstm_out = lstm_out_1+lstm_out_2

if __name__ == '__main__':
    main()
