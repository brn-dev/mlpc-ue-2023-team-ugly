import torch

from lib.data_preprocessing import remove_correlated_columns
from lib.ds.bird_classes import NUM_CLASSES
from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split
from lib.model.classification_transformer import TransformerHyperParameters
from lib.torch_device import get_torch_device
from lib.transformer_training import train_transformer_with_cv

def main():
    device = get_torch_device()

    data_train, labels_train, data_test, labels_test = split(*load_all_data('dataset'), seed=69420)

    # data_train = data_train[:128]

    data_train, data_test = remove_correlated_columns(data_train, data_test)

    hyper_parameters = TransformerHyperParameters(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        out_size=NUM_CLASSES,
        in_features=data_train.shape[-1],
    )

    train_transformer_with_cv(data_train, labels_train, hyper_parameters, device)



if __name__ == '__main__':
    main()
