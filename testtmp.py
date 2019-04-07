from data_pre.alldata import *
import time

# All hyperparameters
n_hidden = 50  # Hidden layer num of features
n_classes = 10  # Total classes (should go up, or should go down)
n_inputs = 10
max_seq = 800


def main():
    time1 = time.time()
    print('loading data...')
    train_sets = CNNData(foldname='./data/actdata/', max_seq=max_seq,
                             num_class=n_classes, trainable=True, kfold_num=0)
    print('load data time:', time.time()-time1)


if __name__ == '__main__':
    main()