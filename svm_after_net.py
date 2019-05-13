'''
SVM 方案在新数据上测试：
main:
    测试网络50 与10 输出作为特征提取的识别结果
main2:
    直接测试使用《软件学报》投稿论文的SVM方法最终结果

'''
from data_pre.alldata import *
import tensorflow as tf
import os
import time
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import svm

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
training_iters = 80  # Loop 1000 times on the dataset
batch_size = 400
display_iter = 2000  # To show test set accuracy during training
model_save = 80

fold = './data/data_svmfea/'

LABELS = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']


def Matrix_to_CSV_array(filename, data):
    with open(filename, "w", newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def main():
    k_fold_num = 0
    time1 = time.time()
    print('loading data...from {}'.format(fold))
    train_sets = data_load_res(foldname=fold, trainable=True, kfold_num=k_fold_num)
    test_sets = data_load_res(foldname=fold, trainable=False, kfold_num=k_fold_num)
    print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
    print('load data time:', time.time()-time1)
    x1 = train_sets.data_res
    y1 = train_sets.all_label.argmax(1)
    x2 = test_sets.data_res
    y2 = test_sets.all_label.argmax(1)

    # c_x_1 = 1.7
    # g_x_1 = 0.05

    # PCA---------------------------------------------------------------------
    # pca_model = PCA(n_components=0.999)
    # pca_model.fit(x3)
    # x3 = pca_model.fit_transform(x3)

    # Ureduce = pca_model.components_.T  # shape(n_components, n_features)
    # x1 = np.dot(x1, Ureduce)
    # x2 = np.dot(x2, Ureduce)
    # # # print('U', Ureduce.T)

    # print('radio', pca_model.explained_variance_ratio_, sum(pca_model.explained_variance_ratio_), len(pca_model.explained_variance_ratio_))
    # print(',', pca_model.explained_variance_, sum(pca_model.explained_variance_))
    # PCA---------------------------------------------------------------------
    cishu = 0
    a_gamma = np.sort(np.array([0.01, 0.02, 0.4, 0.08, 0.005, 0.001, 0.0001, 0.1, 0.5]))
    a_c = np.sort(np.array([1e-3, 1e-2, 1e-1, 0.5, 1, 1.5, 3.0, 5, 10, 100, 1000]))
    # param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    for ks in a_gamma:
        for c_ceshi in a_c:
            cishu += 1
            time2 = time.time()
            c_x_1 = c_ceshi
            g_x_1 = ks
            print('------------------------------------')
            print("SVM training....{}".format(cishu))
            clf_svm = svm.SVC(C=c_x_1, kernel='rbf', gamma=g_x_1, decision_function_shape='ovr')
            clf_svm.fit(x1, y1.ravel())
            jingdu = clf_svm.score(x1, y1)
            quedu = clf_svm.score(x2, y2)
            # print('训练集结果精度', jingdu)
            # print('测试集结果正确率', quedu)
            print("Epoch {},  C:{}   Gamma: {}".format(cishu, c_x_1, g_x_1))
            print('训练集精度：', jingdu, '  测试集准确率：', quedu)

            print("A train epoch time: {:.2f} s\n".format(time.time()-time2))

    # one_hot_predictions, accuracy, final_loss = sess.run(
    #     [pred, accuracy, cost],
    #     feed_dict=feed_dic
    # )
    #
    # Matrix_to_CSV_array('./data/res10/test/fea9_kfold{}'.format(k_fold_num), one_hot_predictions)
    # Matrix_to_CSV_array('./data/res10/label_fea9_kfold{}'.format(k_fold_num), test_sets.all_label)

    # print("FINAL RESULT: " + \
    #       "Batch Loss = {}".format(final_loss) + \
    #       ", Accuracy = {}".format(accuracy))
    # print("All train time = {}".format(time.time() - time1))

    predictions = clf_svm.predict(x2)
    result_labels = y2
    print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
    Matrix_to_CSV_array(filename='./matrix/all_lstm/svm-matrix_kfold{}.txt'.format(k_fold_num), data=confusion_matrix)
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
    plt.savefig('./loss_dir/SVM-Matrix_kfold{}.png'.format(k_fold_num), dpi=300, bbox_inches='tight')


def main2():
    # k_fold_num = 0
    for i in range(5):
        print("Now Kfold:{}---------------------------".format(i))
        k_fold_num = i
        time1 = time.time()
        print('loading data...from {}'.format(fold))
        train_sets = data_load_SVMfeas(foldname=fold, trainable=True, kfold_num=k_fold_num)
        test_sets = data_load_SVMfeas(foldname=fold, trainable=False, kfold_num=k_fold_num)
        print('train:', len(train_sets.all_label), 'test:', len(test_sets.all_label))
        print('load data time:', time.time() - time1)
        x1 = train_sets.data_res
        y1 = train_sets.all_label.argmax(1)
        x2 = test_sets.data_res
        y2 = test_sets.all_label.argmax(1)

        cishu = 0
        a_gamma = np.sort(np.array([0.01, 0.4, 0.001, 0.0001]))
        a_c = np.sort(np.array([1e-3, 1e-2, 1e-1, 0.5, 10, 100, 1000]))
        # param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        for ks in a_gamma:
            for c_ceshi in a_c:
                cishu += 1
                time2 = time.time()
                c_x_1 = c_ceshi
                g_x_1 = ks
                print('------------------------------------')
                print("SVM training....{}".format(cishu))
                clf_svm = svm.SVC(C=c_x_1, kernel='rbf', gamma=g_x_1, decision_function_shape='ovr')
                clf_svm.fit(x1, y1.ravel())
                jingdu = clf_svm.score(x1, y1)
                quedu = clf_svm.score(x2, y2)
                # print('训练集结果精度', jingdu)
                # print('测试集结果正确率', quedu)
                print("Epoch {},  C:{}   Gamma: {}".format(cishu, c_x_1, g_x_1))
                print('训练集精度：', jingdu, '  测试集准确率：', quedu)

                predictions = clf_svm.predict(x2)
                result_labels = y2
                print(
                    "Precision: {}%".format(
                        100 * metrics.precision_score(result_labels, predictions, average="weighted")))
                print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
                print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

                print("")
                print("Confusion Matrix:")
                confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
                print(confusion_matrix)

                print("A train epoch time: {:.2f} s\n".format(time.time() - time2))


if __name__ == '__main__':
    main2()