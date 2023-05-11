import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 400)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 300)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(300, 200)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(200, 150)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(150, 50)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(50, output_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
