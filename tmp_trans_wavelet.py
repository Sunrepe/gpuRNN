'''
采用小波变换去噪,
先变换,去噪后重构原始信号.
'''

import os
import csv
import time
import shutil
import numpy as np
import pywt


def Matrix_to_CSV(filename, data):
    import numpy as np
    data = data.astype('int')
    with open(filename, "a", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8", "label"])
        for row in data:
            writer.writerow(row)


def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def wavelet_trans1(data):
    # data = data.T
    wave_let = pywt.Wavelet('db2')
    data_new = []
    for i in range(8):
        channel_data = data[:, i]

        # 小波变换
        coeffs = pywt.wavedec(channel_data, wavelet=wave_let, level=3)
        new_coeffs = []
        for i_coeffs in coeffs:
            thresh = np.sort(i_coeffs)[int((len(i_coeffs))/2)]/0.6745
            i_coeffs = pywt.threshold(i_coeffs, thresh*3, 'soft', 0)
            new_coeffs.append(i_coeffs)

        # 小波重构
        data_new.append(np.array(pywt.waverec(new_coeffs, wave_let,),'int'))

    data_new = np.array(data_new)
    return data_new.T

            # print(pywt.dwt_max_level(50, wave_let))      # 查看可进行的最高分解层次
        # print(np.array(pywt.waverec(coeffs, 'db5'), 'int'))  # 查看反小波分解
        # print('coffs.shaoe',coeffs[0])


def wavelet_trans(data):
    # data = data.T
    wave_let = pywt.Wavelet('sym4')
    data_new = []
    for i in range(8):
        channel_data = data[:, i]

        # 小波变换
        coeffs = pywt.wavedec(channel_data, wavelet=wave_let, level=3)
        new_coeffs = []
        new_coeffs.append(coeffs[0])
        new_coeffs.append(coeffs[1])
        # 只处理高频部分信号.低频信号保留
        for i in range(2, 4):
            i_coeffs = coeffs[i]
            thresh = np.sort(i_coeffs)[int((len(i_coeffs))/2)]/0.6745
            i_coeffs = pywt.threshold(i_coeffs, thresh*3, 'soft', 0)
            new_coeffs.append(i_coeffs)

        # 小波重构
        data_new.append(np.array(pywt.waverec(new_coeffs, wave_let,), 'int'))

    data_new = np.array(data_new)
    return data_new.T

            # print(pywt.dwt_max_level(50, wave_let))      # 查看可进行的最高分解层次
        # print(np.array(pywt.waverec(coeffs, 'db5'), 'int'))  # 查看反小波分解
        # print('coffs.shaoe',coeffs[0])


def mean_trans(data):
    # data = data.T
    for i in range(data.shape[0]-4):
        data[i, :] = np.mean(data[i:i+4, :], axis=0)
    return data[:-4, :]


def cleanfold(fold):
    if os.path.exists(fold):
        shutil.rmtree(fold)
        os.mkdir(fold)
    else:
        os.mkdir(fold)
    print('Already clean fold!')


def main_datatrans(desfold=''):
    print('transfer data...')
    time1 = time.time()
    foldname = './data/actdata/'
    des_fold = desfold
    cleanfold(des_fold)
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        if oc == 'b.txt':
            print(filename)
            c_filename = des_fold + oa + '_' + ob + '_c.txt'
            csvfile = open(c_filename, "a", newline='')
            c_filewriter = csv.writer(csvfile)

            filename = foldname + filename
            data = Read__mean_2(filename)
            # shutil.copy(foldname + oa + '_' + ob + '_c.txt', '../data/wtdata/')
            cutting = Read__mean_2(foldname + oa + '_' + ob + '_c.txt')
            _last = 0
            for cut in range(0, len(cutting)):
                if cut == 0:
                    tmp_data = data[0:cutting[cut], :]
                else:
                    tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                tmp_data = wavelet_trans1(tmp_data)
                # tmp_data = mean_trans(tmp_data)
                _last += tmp_data.shape[0]
                c_filewriter.writerow([_last])
                Matrix_to_CSV(des_fold + oa + '_' + ob + '_b.txt', tmp_data)
            csvfile.close()
    print('Transfer Time:  {} s'.format(time.time() - time1))

if __name__ == '__main__':
    time1 = time.time()
    # main()
    # print('All Time:  {} s'.format(time.time() - time1))
