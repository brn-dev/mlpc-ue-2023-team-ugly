import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split, create_folds
# from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data


def save_data():
    data_train_old, labels_train_old, data_test_old, labels_test_old = split(*load_all_data('dataset'), seed=7890)
    os.makedirs("np_data", exist_ok=True)

    np.save(os.path.join('np_data','data_train.npy'), data_train_old)
    np.save(os.path.join('np_data','labels_train.npy'), labels_train_old)
    np.save(os.path.join('np_data','data_test.npy'), data_test_old)
    np.save(os.path.join('np_data','labels_test.npy'), labels_test_old)

    data_train = np.load(os.path.join('np_data','data_train.npy'))
    labels_train = np.load(os.path.join('np_data','labels_train.npy'))
    data_test = np.load(os.path.join('np_data','data_test.npy'))
    labels_test = np.load(os.path.join('np_data','labels_test.npy'))

    print(f'{data_train.shape}')
    print(np.all(data_train == data_train_old))
    print(labels_train.shape)
    print(np.all(labels_train == labels_train_old))
    print(data_test.shape)
    print(np.all(data_test == data_test_old))
    print(labels_test.shape)
    print(np.all(labels_test == labels_test_old))

    data_train_folds_old, labels_train_folds_old = create_folds(data_train, labels_train, n_folds=10)
    np.save(os.path.join('np_data','data_train_folds.npy'), data_train_folds_old)
    np.save(os.path.join('np_data','labels_train_folds.npy'), labels_train_folds_old)

    data_train_folds = np.load(os.path.join('np_data','data_train_folds.npy'))
    labels_train_folds = np.load(os.path.join('np_data','labels_train_folds.npy'))

    print(data_train_folds.shape)
    print(np.all(data_train_folds==data_train_folds_old))

    print(labels_train_folds.shape)
    print(np.all(labels_train_folds == labels_train_folds_old))


def save_data_2():
    data_train = np.load(os.path.join('np_data','data_train.npy'))
    labels_train = np.load(os.path.join('np_data','labels_train.npy'))
    data_test = np.load(os.path.join('np_data','data_test.npy'))

    print(data_train.shape, data_test.shape)
    data_train_down_old, data_test_down_old = remove_correlated_columns(data_train, data_test)

    np.save(os.path.join('np_data','data_train_down.npy'), data_train_down_old)
    np.save(os.path.join('np_data','data_test_down.npy'), data_test_down_old)

    data_train_down, data_test_down = np.load(os.path.join('np_data','data_train_down.npy')), np.load(os.path.join('np_data','data_test_down.npy'))
    print(data_train_down.shape, data_test_down.shape)
    print(np.all(data_train_down==data_train_down_old))
    print(np.all(data_test_down==data_test_down_old))


    ## Creating the folds with the down projected data

    data_train_folds_down_old, labels_train_folds_down_old = create_folds(data_train_down, labels_train, n_folds=10)
    np.save(os.path.join('np_data','data_train_folds_down.npy'), data_train_folds_down_old)
    np.save(os.path.join('np_data','labels_train_folds_down.npy'), labels_train_folds_down_old)

    data_train_folds_down = np.load(os.path.join('np_data','data_train_folds_down.npy'))
    labels_train_folds_down = np.load(os.path.join('np_data','labels_train_folds_down.npy'))

    print(data_train_folds_down.shape)
    print(np.all(data_train_folds_down==data_train_folds_down_old))

    print(labels_train_folds_down.shape)
    print(np.all(labels_train_folds_down == labels_train_folds_down_old))


if __name__ == '__main__':
    save_data()
    save_data_2()
    pass