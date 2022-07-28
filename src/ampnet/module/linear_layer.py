import torch
from torch.nn import Linear

class LinearLayer(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.lin1 = Linear(in_features=2, out_features=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [num_samples, 2]
        x = self.lin1(x)
        return x
