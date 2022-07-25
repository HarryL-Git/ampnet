import os
import math
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN
from synthetic_benchmark.synthetic_xor import create_data


# Global variables
TRAIN_AMPCONV = False  # If False, trains a simple 2-layer GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create save paths
save_path = "./synthetic_benchmark/runs" if TRAIN_AMPCONV else "./synthetic_benchmark/runs_GCN"
# save_path = "./synthetic_benchmark/runs_linear_layer"
if not os.path.exists(save_path):
    os.mkdir(save_path)

SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
    os.system("cp ./synthetic_benchmark/synthetic_training.py {}/".format(SAVE_PATH))

if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)
if not os.path.exists(ACTIV_PATH):
    os.mkdir(ACTIV_PATH)


class LinearLayer(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.lin1 = nn.Linear(in_features=2, out_features=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [num_samples, 2]
        x = self.lin1(x)
        return x


class TwoLayerSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Running Two Layer MLP with Sigmoid Activation")
        self.lin1 = nn.Linear(in_features=2, out_features=2)
        self.act1 = nn.Sigmoid()
        self.lin2 = nn.Linear(in_features=2, out_features=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [num_samples, 2]
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x


if TRAIN_AMPCONV:
    model = AMPGCN().to(device)
else:
    model = GCN(num_node_features=2, output_dim=1, softmax_out=False).to(device)
# model = TwoLayerSigmoid()
# model = LinearLayer()

# Define data
x, y, adj_matrix, edge_idx_arr = create_data(num_samples=20, noise_std=0.05, same_class_link_prob=0.7, diff_class_link_prob=0.1)
train_data = Data(x=x, edge_index=edge_idx_arr, y=y)
x, y, adj_matrix, edge_idx_arr = create_data(num_samples=20, noise_std=0.05, same_class_link_prob=0.7, diff_class_link_prob=0.1)
test_data = Data(x=x, edge_index=edge_idx_arr, y=y)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(train_data)
    # train_loss = F.nll_loss(out, train_data.y)
    # train_loss = 0.5 * (out - train_data.y) ** 2
    # train_loss = F.mse_loss(out.squeeze(-1), train_data.y)
    train_loss = criterion(out.squeeze(-1), train_data.y)
    pred = (out.squeeze(-1) > 0.5).float()
    train_accuracy = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())

    train_loss.backward()
    # if epoch % 4 == 0:
    #     model.plot_grad_flow(GRADS_PATH, epoch, idx=0)
    #     model.visualize_gradients(GRADS_PATH, epoch, idx=0)
    #     model.visualize_activations(ACTIV_PATH, data, epoch, idx=0)
    optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        # test_loss = F.nll_loss(out, test_data.y)
        # test_loss = F.mse_loss(out, test_data.y)
        pred = (out.squeeze(-1) > 0.5).float()
        test_loss = criterion(out.squeeze(-1), test_data.y)
        test_accuracy = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())
    
    print("Epoch {:05d} | Train Loss {:.4f}; Acc {:.4f} | Test Loss {:.4f} | Acc {:.4f} "
            .format(epoch, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
    train_loss_list.append(train_loss.item())
    train_acc_list.append(train_accuracy)
    test_loss_list.append(test_loss.item())
    test_acc_list.append(test_accuracy)


print("Training took {} minutes.".format((time.time() - start_time) / 60.))
# plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="2LayerSigmoid")
# plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="2LayerSigmoid")
if TRAIN_AMPCONV:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="AMPConv")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="AMPConv")
else:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="GCN")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="GCN")

model.eval()
with torch.no_grad():
    pred = model(test_data).argmax(dim=1)
    correct = (pred == test_data.y).sum()
    acc = int(correct) / test_data.x.shape[0]
print(f'Final Test Accuracy: {acc:.4f}')
