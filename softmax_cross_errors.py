import tensorflow as tf
import numpy as np
import csv
import time
import scipy.stats as ss


def Read_data_res(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 组合的数据
    '''
    my_matrix = np.loadtxt(filename, dtype='float', delimiter=",")
    return my_matrix


def Matrix_to_CSV(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


class getAlldata_cross_errors(object):
    def __init__(self, n_class=10, kfold_num=0):
        self.alldata = []
        for fea_n in range(9):
            filename = './data/res{}/train/fea{}_kfold{}'.format(n_class, fea_n, kfold_num)
            self.alldata.append(Read_data_res(filename))

    def get_data(self, fea_num=0):
        return self.alldata[fea_num]



def softmax(x):
    # 1
    shift_x = x - np.max(x)    # 防止输入增大时输出为nan
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

    # 2
    # exp_x = np.exp(x)
    # return exp_x / np.sum(exp_x)



def main():
    cs_errors_matrix = np.zeros([9, 9])
    num_class = 50
    # kfold_num = 0
    time0 = time.time()

    x = tf.placeholder(tf.float32, [None, num_class])
    y = tf.placeholder(tf.float32, [None, num_class])

    # y_ = tf.nn.softmax(y)
    # x_ = tf.nn.softmax(x)
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(x_)))

    # alpha = tf.reduce_max(logits, axis=-1, keepdims=True)
    # log_sum_exp = tf.log(tf.reduce_sum(tf.exp(logits - alpha), axis=-1, keepdims=True)) + alpha
    # cross_entropy = -tf.reduce_sum((logits - log_sum_exp) * labels, axis=-1)
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # alpha = tf.reduce_max(y, axis=-1, keepdims=True)
    # y_label = tf.log(tf.reduce_sum(tf.exp(y - alpha), axis=-1, keepdims=True)) + alpha

    # time1 = time.time()
    # data = getAlldata_cross_errors(num_class, kfold_num)
    # print('load data time:', time.time() - time1)

    # do cross_entropy just one step
    # tf.nn.sparse_softmax_cross_entropy_with_logits()
    cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x))

    with tf.Session() as sess:
        for kfold_num in range(5):
            print('Loading data...')
            print("K_fold{}".format(kfold_num))
            time1 = time.time()
            data = getAlldata_cross_errors(num_class, kfold_num)
            print('load data time:', time.time() - time1)

            for i_fea in range(9):
                res_y = softmax(data.get_data(i_fea))
                # res_y = data.get_data(i_fea)
                for i_next in range(i_fea, 9):
                    res_x = data.get_data(i_next)
                    feed_dict = {
                        x: res_x,
                        y: res_y
                    }
                    c_e2 = sess.run(cross_entropy2, feed_dict=feed_dict)
                    cs_errors_matrix[i_fea, i_next] = c_e2
                    cs_errors_matrix[i_next, i_fea] = c_e2
                    print('fea:{}--{}:\t{:.6f}'.format(i_fea, i_next, c_e2))

            Matrix_to_CSV(filename='./data/cs_errors/res{}_kfold{}.csv'.format(num_class, kfold_num), data=cs_errors_matrix)

    print('All time:', time.time() - time0)


def main2():
    cs_errors_matrix = np.zeros([9, 9])
    num_class = 10
    time0 = time.time()

    for kfold_num in range(5):
        print('Loading data...')
        print("K_fold{}".format(kfold_num))
        time1 = time.time()
        data = getAlldata_cross_errors(num_class, kfold_num)
        print('load data time:', time.time() - time1)

        for i_fea in range(9):
            res_y = softmax(data.get_data(i_fea))
            # res_y = data.get_data(i_fea)
            for i_next in range(i_fea, 9):
                res_x = softmax(data.get_data(i_next))
                c_e2 = ss.entropy(res_x.T, res_y.T)
                c_e2 = np.mean(c_e2)
                # c_e2 = sess.run(cross_entropy2, feed_dict=feed_dict)
                cs_errors_matrix[i_fea, i_next] = c_e2
                cs_errors_matrix[i_next, i_fea] = c_e2
                print('fea:{}--{}:\t{:.6f}'.format(i_fea, i_next, c_e2))

        Matrix_to_CSV(filename='./data/cs_errors/2res{}_kfold{}.csv'.format(num_class, kfold_num),
                      data=cs_errors_matrix)

    print('All time:', time.time() - time0)


if __name__ == '__main__':
    main2()