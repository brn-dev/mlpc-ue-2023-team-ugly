import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split, create_folds
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# global CUDA_LAUNCH_BLOCKING
# CUDA_LAUNCH_BLOCKING = 1

def main():
    # x = load_all_data('dataset')[1].max(axis=1)
    # print(x.shape)
    # print(x)

    data_train_raw, labels_train, data_test_raw, labels_test = split(*load_all_data('dataset'), seed=69421)

    create_folds(data_train_raw, labels_train, 10, None)


if __name__ == '__main__':
    main()

