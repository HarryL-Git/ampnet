#!/usr/bin/env python
#
# Intended as a simple benchmark of the AMPNetClassifier against the Cora data set
#
# @author Rahul Dhodapkar

import time
from tracemalloc import start
from src.ampnet.utils.preprocess import embed_features  # , embed_features_2, embed_features_3


from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from src.ampnet.conv.amp_conv import AMPConv
from torch_geometric.loader import GraphSAINTRandomWalkSampler


def accuracy(v1, v2):
    return (v1 == v2).sum() / v1.shape[0]


# class AMPGCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()

#         self.conv1 = AMPConv(
#             embed_dim=10,
#             num_heads=2,
#             first_amp_conv=True
#         )
#         self.conv2 = AMPConv(
#             embed_dim=10,
#             num_heads=2,
#             first_amp_conv=False
#         )
#         # self.convs.append(self.conv1)
#         self.lin1 = nn.Linear(in_features=14330, out_features=dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index  # x becomes [num_nodes, 1433]
#         x = embed_features(x, feature_embed_dim=4, value_embed_dim=2)  # x becomes [300, 8598]

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         x = self.lin1(x)
#         return F.log_softmax(x, dim=1)
    
    # def forward(self, x, adjs):
    #     # Forward function code based on tutorial from PyTorch Geometric for handling neighbor sampling dataloader
    #     for i, (edge_index, _, size) in enumerate(adjs):
    #         x_target = x[:size[1]]  # Target nodes are always placed first.
    #         x = self.convs[i]((x, x_target), edge_index)

    #         x = F.relu(x)
    #         x = F.dropout(x, training=self.training)

    #     return F.log_softmax(x, dim=1)
    

# class AMPGCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()

#         self.conv1 = AMPConv(in_channels=1433, out_channels=16)
#         self.conv2 = AMPConv(in_channels=16, out_channels=dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index  # x becomes [num_nodes, 1433]
#         # x = embed_features(x, feature_embed_dim=4, value_embed_dim=2)  # x becomes [300, 8598]

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)

    
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 16)  # dataset.num_node_features 
        self.conv2 = GCNConv(16, dataset.num_classes)
        # self.lin0 = nn.Linear(1433, 600)
        # self.lin1 = nn.Linear(600, 1433)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x becomes [num_nodes, 1433]
        # x = embed_features(x, feature_embed_dim=4, value_embed_dim=2)  # x becomes [num_nodes, 8598]
        # x = self.lin0(x)
        # x = self.lin1(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0].to(device)
model = GCN().to(device)
# model = AMPGCN().to(device)

loader = GraphSAINTRandomWalkSampler(all_data, batch_size=100, walk_length=3,
                                     num_steps=10, sample_coverage=100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
start_time = time.time()

"""
Notes: GraphSAINT RandomWalkSampler
- GCN with num_steps=10, batch_size=120, walk_length=2: 78% accuracy, 170 MB memory
- GCN with num_steps=10, batch_size=120, walk_length=3: 79% accuracy, 170 MB memory
- GCN with num_steps=10, batch_size=100, walk_length=3: 81% accuracy, 170 MB memory
- AMPGCN with num_steps=10, batch_size=120, walk_length=3: 
- 
38 GB"""

for epoch in range(20):
    model.train()

    total_loss = total_examples = 0
    for idx, data_obj in enumerate(loader):
        data = data_obj.to(device)
        optimizer.zero_grad()

        # edge_weight = data.edge_norm * data.edge_weight
        out = model(data)
        train_loss = F.nll_loss(out, data.y, reduction='none')  # Wrong, no test mask!
        train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
        train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                    data.y[data.train_mask].detach().numpy())
        
        test_loss = F.nll_loss(out, data.y, reduction='none')  # Wrong, no test mask!
        test_loss = (test_loss * data.node_norm)[data.test_mask].sum()
        test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).numpy(),
                                    data.y[data.test_mask].detach().numpy())

        print("Epoch {:05d} Partition {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} "
                .format(epoch, idx, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))

        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item() * data.num_nodes
        total_examples += data.num_nodes


print("Training took {} minutes.".format((time.time() - start_time) / 60.))

model.eval()
pred = model(all_data).argmax(dim=1)
correct = (pred[all_data.test_mask] == all_data.y[all_data.test_mask]).sum()
acc = int(correct) / int(all_data.test_mask.sum())
print(f'Final Test Accuracy: {acc:.4f}')
