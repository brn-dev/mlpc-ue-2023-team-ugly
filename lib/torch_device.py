import torch

PREFERRED = None

def get_torch_device() -> torch.device:
    device = torch.device(PREFERRED if PREFERRED is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    return device
