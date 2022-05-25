#!/usr/bin/env python
#
# Create synthetic benchmark dataset for AMPConv vs other architectures
#
# @author Rahul Dhodapkar

import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch.nn import Linear

from src.ampnet.conv.amp_conv import AMPConv

# for reproducibility
np.random.seed(seed=1)

# create several channels that are informative, and several channels that are uninformative
def create_two_sample_dataset(n_class_1=25,
                              n_class_2=25,
                              mean_1=0.7,
                              mean_2=0,
                              n_informative_features=50,
                              n_noise_features=50,
                              homotypic_edge_prob=0.8,
                              heterotypic_edge_prob=0.3):
    y = np.concatenate((np.repeat(0, n_class_1), np.repeat(1, n_class_2)))
    class_1_ixs = list(range(0, n_class_1))
    class_2_ixs = list(range(n_class_1, n_class_1 + n_class_2))

    informative_x = np.concatenate((
        np.random.normal(loc=mean_1, scale=1, size=(n_class_1, n_informative_features)),
        np.random.normal(loc=mean_2, scale=1, size=(n_class_2, n_informative_features))
    ), axis=0)
    noise_x = np.concatenate((
        np.random.normal(loc=0, scale=1, size=(n_class_1, n_noise_features)),
        np.random.normal(loc=0, scale=1, size=(n_class_2, n_noise_features))
    ), axis=0)
    x = np.concatenate((informative_x, noise_x), axis=1)

    edges = []
    for i in range(n_class_1 + n_class_2):
        for j in range(n_class_1 + n_class_2):
            if y[i] == y[j] and np.random.random() < homotypic_edge_prob:
                edges.append((i, j))
            elif y[i] != y[j] and np.random.random() < heterotypic_edge_prob:
                edges.append((i, j))
    edge_index = torch.tensor(np.array(edges), dtype=torch.long)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.int64),
        edge_index=edge_index.t().contiguous(),
        num_classes=2)


data = create_two_sample_dataset()
transform = T.RandomNodeSplit(
        num_train_per_class=20,
        num_val=20,
        num_test=20
    )
data = transform(data)


class AMPGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Linear(data.x.shape[1], data.x.shape[1] * 3)
        self.conv1 = AMPConv(embed_dim=3, num_heads=1)
        self.conv2 = AMPConv(embed_dim=3, num_heads=1)
        self.linear = Linear(data.x.shape[1] * 3, np.max(data.y.numpy()) + 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embed(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, np.max(data.y.numpy()) + 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AMPGCN().to(device)
#model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    print("epoch {}".format(epoch))
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

