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
save_path = "./synthetic_benchmark/runs" if TRAIN_AMPCONV else "./synthetic_benchmark/runs_GCN_baseline"
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


if TRAIN_AMPCONV:
    model = AMPGCN().to(device)
else:
    model = GCN().to(device)

# Define data
x, y, adj_matrix, edge_idx_arr = create_data(num_samples=20)
data = Data(x=x, edge_index=edge_idx_arr, y=y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                data.y[data.train_mask].detach().numpy())

    train_loss.backward()
    if epoch % 4 == 0:
        model.plot_grad_flow(GRADS_PATH, epoch, idx=0)
        model.visualize_gradients(GRADS_PATH, epoch, idx=0)
        model.visualize_activations(ACTIV_PATH, data, epoch, idx=0)
    optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).numpy(),
                                    data.y[data.test_mask].detach().numpy())
    
    print("Epoch {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f} | Acc {:.4f} "
            .format(epoch, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
    train_loss_list.append(train_loss.item())
    train_acc_list.append(train_accuracy)
    test_loss_list.append(test_loss.item())
    test_acc_list.append(test_accuracy)


print("Training took {} minutes.".format((time.time() - start_time) / 60.))
if TRAIN_AMPCONV:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="AMPConv")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="AMPConv")
else:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="GCN")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="GCN")

model.eval()
with torch.no_grad():
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
print(f'Final Test Accuracy: {acc:.4f}')
