import numpy as np
import os

from lib.ds.dataset_loading import load_all_data, flatten
from lib.ds.dataset_splitting import split, create_folds, split_by_1d, redistribute_labels
from lib.data_preprocessing import remove_correlated_columns, find_best_features

data_path = 'np_data_new'

os.makedirs(data_path, exist_ok=True)


def save_data():
    data_orig, labels_orig = load_all_data('dataset')

    data_train_old, labels_train_old, data_test_old, labels_test_old = split_by_1d(data_orig, labels_orig)


    np.save(os.path.join(data_path, 'data_train.npy'), data_train_old)
    np.save(os.path.join(data_path,'labels_train.npy'), labels_train_old)
    np.save(os.path.join(data_path,'data_test.npy'), data_test_old)
    np.save(os.path.join(data_path,'labels_test.npy'), labels_test_old)
    #
    data_train = np.load(os.path.join(data_path,'data_train.npy'))
    labels_train = np.load(os.path.join(data_path,'labels_train.npy'))
    data_test = np.load(os.path.join(data_path,'data_test.npy'))
    labels_test = np.load(os.path.join(data_path,'labels_test.npy'))

    #
    print(f'{data_train.shape}')
    print(np.all(data_train == data_train_old))
    print(labels_train.shape)
    print(np.all(labels_train == labels_train_old))
    print(data_test.shape)
    print(np.all(data_test == data_test_old))
    print(labels_test.shape)
    print(np.all(labels_test == labels_test_old))

    data_train_down, data_test_down = remove_correlated_columns(data_train, data_test)

    np.save(os.path.join(data_path, 'data_train_down.npy'), data_train_down)
    np.save(os.path.join(data_path, 'data_test_down.npy'), data_test_down)

    print(data_train_down.shape)
    print(data_test_down.shape)

    balanced_data_train, balanced_labels_train = redistribute_labels(data_train_down, labels_train)
    balanced_data_train, balanced_labels_train = flatten(balanced_data_train, balanced_labels_train)
    flatten_data_test, flatten_labels_test = flatten(data_test_down, labels_test)

    np.save(os.path.join(data_path, 'balanced_data_train_down.npy'), balanced_data_train)
    np.save(os.path.join(data_path, 'balanced_labels_train.npy'), balanced_labels_train)

    np.save(os.path.join(data_path, 'flatten_data_test.npy'),flatten_data_test)
    np.save(os.path.join(data_path, 'flatten_labels_test.npy'), flatten_labels_test)

    print(balanced_data_train.shape)
    print(balanced_labels_train.shape)
    print(flatten_data_test.shape)
    print(flatten_labels_test.shape)


def save_data_2():
    data_train = np.load(os.path.join('np_data','data_train.npy'))
    labels_train = np.load(os.path.join('np_data','labels_train.npy'))
    data_test = np.load(os.path.join('np_data','data_test.npy'))

    print(data_train.shape, data_test.shape)
    data_train_down_old, data_test_down_old = remove_correlated_columns(data_train, data_test)
    data_train_down_old, data_test_down_old = find_best_features(data_train_down_old, labels_train, data_test_down_old)

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
    pass