import os
import math
import time
import random
import datetime

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN

os.chdir("..")  # Change current working directory to parent directory of GitHub repository
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# Global variables
TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0]


# Create save paths
save_path = "experiments/runs" if TRAIN_AMPCONV else "experiments/runs_GCN_baseline"
if not os.path.exists(save_path):
    os.mkdir(save_path)

SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
    os.system("cp experiments/cora_benchmark_graphsaint.py {}/".format(SAVE_PATH))
    # if TRAIN_AMPCONV:
    #     os.system("cp src/ampnet/conv/amp_conv.py {}/".format(SAVE_PATH))
    #     os.system("cp src/ampnet/module/amp_gcn.py {}/".format(SAVE_PATH))
    # else:
    #     os.system("cp src/ampnet/module/gcn_classifier.py {}/".format(SAVE_PATH))

if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)
if not os.path.exists(ACTIV_PATH):
    os.mkdir(ACTIV_PATH)


if TRAIN_AMPCONV:
    model = AMPGCN(
            device=device,
            embedding_dim=128, 
            num_heads=4,
            num_node_features=1433,
            num_sampled_vectors=20,
            output_dim=7,
            softmax_out=True, 
            feat_emb_dim=127, 
            val_emb_dim=1,
            downsample_feature_vectors=True,
            average_pooling_flag=True,
            dropout_rate=0.0,
            dropout_adj_rate=0.0,
            feature_repeats=None).to(device)
else:
    model = GCN().to(device)

# batch size 1, walk length 500 => ~225 nodes
# batch size 20, walk length 100 => ~750 nodes
# batch size 10, walk length 100 => ~500 nodes
num_steps = 200
loader = GraphSAINTRandomWalkSampler(all_data, batch_size=8, walk_length=150,
                                     num_steps=num_steps, sample_coverage=100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=2)
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 10000 // num_steps
for epoch in range(epochs):

    total_loss = total_examples = 0
    for idx, data_obj in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        test_loss = 0.0
        data = data_obj.to(device)

        # edge_weight = data.edge_norm * data.edge_weight
        out = model(data)
        train_loss = F.nll_loss(out, data.y.to(device), reduction='none')
        train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
        train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).cpu().numpy(),
                                    data.y[data.train_mask].detach().cpu().numpy())
        
        train_loss.backward()
        if idx % 4 == 0:
            model.plot_grad_flow(GRADS_PATH, epoch, idx)
            model.visualize_gradients(GRADS_PATH, epoch, idx)
            model.visualize_activations(ACTIV_PATH, data, epoch, idx)
        optimizer.step()
        scheduler.step()
        total_loss += train_loss.item() * data.num_nodes
        total_examples += data.num_nodes

        ########
        # Test #
        ########
        model.eval()
        with torch.no_grad():
            test_loss = F.nll_loss(out, data.y.to(device), reduction='none')
            test_loss = (test_loss * data.node_norm)[data.test_mask].sum()
            test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).cpu().numpy(),
                                        data.y[data.test_mask].detach().cpu().numpy())

        print("Epoch {:05d} Partition {:05d} LR {:.05f} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} "
                .format(epoch, idx, float(scheduler.get_last_lr()[0]), train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_accuracy)
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_accuracy)

    # Save model checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'validation_loss': test_loss.item()
        }, os.path.join(SAVE_PATH, "model_checkpoint_ep{}.pth".format(epoch)))

print("Training took {} minutes.".format((time.time() - start_time) / 60.))
if TRAIN_AMPCONV:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="AMPConv")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="AMPConv")
else:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="GCN")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="GCN")

torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'validation_loss': test_loss.item()
}, os.path.join(SAVE_PATH, "final_model.pth"))

model.eval()
with torch.no_grad():
    pred = model(all_data).argmax(dim=1)
    correct = (pred[all_data.test_mask].cpu() == all_data.y[all_data.test_mask].cpu()).sum()
    acc = int(correct) / int(all_data.test_mask.cpu().sum())
print(f'Final Test Accuracy: {acc:.4f}')
