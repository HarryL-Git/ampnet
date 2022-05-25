#!/usr/bin/env python
#
# Intended as a simple benchmark of the AMPNetClassifier against the Cora data set
#
# @author Rahul Dhodapkar

import numpy as np

from src.ampnet.utils.preprocess import embed_features

from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.ampnet.conv.amp_conv import AMPConv
from torch_geometric.loader import GraphSAINTRandomWalkSampler


############################
def accuracy(v1, v2):
    return (v1 == v2).sum() / v1.shape[0]

############################

class AMPGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = AMPConv(
            embed_dim=12,
            num_heads=3
        )
        self.conv2 = AMPConv(
            embed_dim=12,
            num_heads=3
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = embed_features(x, feature_embed_dim=8, value_embed_dim=4)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GCN().to(device)
model = AMPGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

dataset = Planetoid(root='/tmp/Cora', name='Cora')

model.train()
for epoch in range(5):
    loader = GraphSAINTRandomWalkSampler(dataset[0], batch_size=20, walk_length=2,
                                         num_steps=5, sample_coverage=100)

    partition_ix = 0
    for data_obj in loader:
        data = data_obj.to(device)

        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(),
                                  data.y[data.train_mask].detach().numpy())

        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).numpy(),
                                 data.y[data.test_mask].detach().numpy())

        print("Epoch {:05d} Partition {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} ".format(
            epoch, partition_ix
            , train_loss.item(), train_accuracy
            , test_loss.item(), test_accuracy))

        train_loss.backward()
        optimizer.step()
        partition_ix = partition_ix + 1


data = dataset[0]
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
