import torch
import math


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data = torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)

class MinimalisticNetwork(torch.nn.Module):
    '''
    This is a simple network for DeepSurv using Dense Layers.
    '''

    def __init__(self, input_dim=15, inner_dim=128) -> None:
        '''
        Initialize the network with the input and output dimensions.
        @param input_dim: The input dimension of the network.
        @param inner_dim: The inner dimension of the network.
        '''
        super().__init__()
        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(input_dim, inner_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(inner_dim),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(inner_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, x, *args, **kwargs):
        '''
        Forward pass of the network.
        @param x: The input tensor.
        @return: The output tensor.
        '''
        x = self.network(x)

        return x 