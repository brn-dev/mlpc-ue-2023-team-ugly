import torch

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split
from lib.model.classification_transformer import TransformerHyperParameters
from lib.torch_device import get_torch_device
from lib.transformer_training import train_transformer_with_cv

def main():
    device = get_torch_device()

    data_train, labels_train, data_test, labels_test = split(*load_all_data('dataset'), seed=69420)

    hyper_parameters = TransformerHyperParameters(
        in_features=data_train.shape[-1]
    )

    train_transformer_with_cv(data_train, labels_train, hyper_parameters, device)



if __name__ == '__main__':
    main()
