from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split

def main():
    # x = load_all_data('dataset')[1].max(axis=1)
    # print(x.shape)
    # print(x)

    data_train, labels_train, data_test, labels_test = split(*load_all_data('dataset'))
    print(data_train.shape)
    print(data_train)
    print(labels_train.shape)
    print(labels_train)
    print(data_test.shape)
    print(data_test)
    print(labels_test.shape)
    print(labels_test)


if __name__ == '__main__':
    main()

