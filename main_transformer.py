import torch

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split
from lib.transformer import TransformerHyperParameters, train_transformer_with_cv

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_train, labels_train, data_test, labels_test = split(*load_all_data('dataset'), seed=69420)

    hyper_parameters = TransformerHyperParameters()

    train_transformer_with_cv(data_train, labels_train, hyper_parameters, device)



if __name__ == '__main__':
    main()
