import os
import time
import datetime

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN


TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)

if not os.path.exists("./runs"):
    os.mkdir("./runs")

SAVE_PATH = os.path.join("./runs_full_graph", datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)


if TRAIN_AMPCONV:
    model = AMPGCN().to(device)
else:
    model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
start_time = time.time()

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 200
for epoch in range(epochs):
    model.train()
    model.training = True
    optimizer.zero_grad()

    out = model(data)
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                data.y[data.train_mask].detach().numpy())

    train_loss.backward()
    if epoch % 10 == 0:
        model.plot_grad_flow(GRADS_PATH, epoch)
        model.visualize_gradients(GRADS_PATH, epoch)
    optimizer.step()

    # Test
    model.eval()
    model.training = False
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
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
