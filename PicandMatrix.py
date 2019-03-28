import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import os

def PIC_loss_and_acc(train_loss,train_acc,test_loss,test_acc,display_iter,training_iters,batch_size):

    # PIC_loss_and_acc(train_losses, train_accuracies, test_losses, test_accuracies, display_item, training_iters * 2560,batch_size)
    font = {
        'family': 'Bitstream Vera Sans',
        'weight': 'bold',
        'size': 18
    }
    matplotlib.rc('font', **font)

    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_loss)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_loss), "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_acc), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_loss)*display_iter, display_iter)[:-1]),
        [training_iters*len(test_losses)]
    )
    plt.plot(indep_test_axis, np.array(test_loss), "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_acc), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()


def PICforConfusionMatrix(predictions,result_labels,num_classes, _Labels=None,):
    '''
    Get confusion matrix according to predictions and labels.

    :param predictions: The Net pred result
    :param result_labels: The true label (Not one-hot)
    :param num_classes:
    :param _Labels: Use for create x/y axis
    :return:
    '''
    print("")
    print("Precision: {}%".format(100 * metrics.precision_score(result_labels, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(result_labels, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(result_labels, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(result_labels, predictions)
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
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, _Labels, rotation=90)
    plt.yticks(tick_marks, _Labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    train_losses = np.loadtxt('./loss3/train_loss.txt')
    train_accuracies = np.loadtxt('./loss3/train_acc.txt')
    test_losses = np.loadtxt('./loss3/test_loss.txt')
    test_accuracies = np.loadtxt('./loss3/test_acc.txt')

    training_iters = 200  # Loop 1000 times on the dataset
    batch_size = 100
    display_iter = 3000  # To show test set accuracy during training

    PIC_loss_and_acc(train_losses,train_accuracies,test_losses,test_accuracies,display_iter,training_iters,batch_size)