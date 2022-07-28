import torch
from torch.nn import Linear, Sigmoid


class TwoLayerSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Running Two Layer MLP with Sigmoid Activation")
        self.lin1 = Linear(in_features=2, out_features=2)
        self.act1 = Sigmoid()
        self.lin2 = Linear(in_features=2, out_features=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [num_samples, 2]
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x
