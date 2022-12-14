import os
import math
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from src.ampnet.utils.preprocess import embed_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from src.ampnet.conv.amp_conv import AMPConv
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN

"""
This script is out of date, and needs to be refactored before running again.
"""

os.chdir("..")  # Change current working directory to parent directory of GitHub repository


# Global variables
TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN
SAVE_PATH = "./experiments/runs_overfit_one_subgraph"


class AMPGCN(torch.nn.Module):
    def __init__(self, embedding_dim=768, num_sampled_vectors=20, average_pooling_flag=True):
        super().__init__()
        self.emb_dim = embedding_dim
        self.num_sampled_vectors = num_sampled_vectors
        self.average_pooling_flag = average_pooling_flag
        # channels = 20 * 768 if average_pooling_flag else 21 * 768
        if not average_pooling_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.initialize_weights()

        self.conv1 = AMPConv(embed_dim=self.emb_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.act1 = nn.ReLU()
        # self.drop1 = nn.Dropout(p=0.5)

        self.conv2 = AMPConv(embed_dim=self.emb_dim, num_heads=1)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.act2 = nn.ReLU()
        # self.drop2 = nn.Dropout(p=0.5)

        self.conv3 = AMPConv(embed_dim=self.emb_dim, num_heads=1)
        self.norm3 = nn.LayerNorm(self.emb_dim)
        self.act3 = nn.ReLU()

        self.lin1 = nn.Linear(self.emb_dim, out_features=dataset.num_classes)
        # self.lin2 = nn.Linear(in_features=16, out_features=dataset.num_classes)


    def initialize_weights(self):
        nn.init.normal_(self.cls_token, std=.02)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x becomes [num_nodes, 1433]
        x = embed_features(x, feature_emb_dim=self.emb_dim, value_emb_dim=0)  # x becomes [num_nodes, 1000]
        
        # Append class token
        if not self.average_pooling_flag:
            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # x -> [num_nodes]
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = torch.reshape(x, (x.shape[0], (self.num_sampled_vectors + 1) * self.emb_dim))

        x = self.conv1(x, edge_index)
        x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
        x = self.norm1(x)
        x = self.act1(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))
        # x = self.drop1(x)

        x = self.conv2(x, edge_index)
        x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
        x = self.norm2(x)
        x = self.act2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))
        # x = self.drop2(x)

        x = self.conv3(x, edge_index)
        x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
        x = self.norm3(x)
        x = self.act3(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))

        # Reshape node features into unrolled list of feature vectors, perform average pooling
        x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
        if self.average_pooling_flag:
            x = x.mean(dim=1)  # Average pooling
        else:
            x = x[:,0]

        x = self.lin1(x)
        # x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
    def visualize_gradients(self, save_path, epoch_idx, color="C0"):
        """
        Visualization code partly taken from the notebook tutorial at
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
        """
        grads = {name: params.grad.data.view(-1) for name, params in list(self.named_parameters()) if "weight" in name and params.grad is not None}  # Only seeing gradients for leaf nodes, non-leaves grad is None

        gradient_distrib_save_path = os.path.join(save_path, "gradient_distrib_plots")
        if not os.path.exists(gradient_distrib_save_path):
            os.mkdir(gradient_distrib_save_path)
        
        # columns = len(grads)
        # fig, ax = plt.subplots(1, columns, figsize=(columns*4, 4))
        columns = 6
        rows = math.ceil(len(grads)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in grads:
            key_ax = ax[fig_index//columns][fig_index%columns]
            sns.histplot(data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
            mean = grads[key].mean()
            median = grads[key].median()
            mode = grads[key].flatten().mode(dim=0)[0].item()
            std = grads[key].std()
            key_ax.set_title(str(key) + "\nMean: {:.4f}, Median: {:.4f}\nMode: {:.4f}, STD: {:.4f}".format(mean, median, mode, std))
            key_ax.set_xlabel("Grad magnitude")
            fig_index += 1
        fig.suptitle(f"Gradient Magnitude Distribution", fontsize=14, y=1.05)
        fig.subplots_adjust(wspace=0.45)
        plt.tight_layout()
        plt.savefig(os.path.join(gradient_distrib_save_path, "gradient_distrib_epoch{}".format(epoch_idx)), bbox_inches='tight', facecolor="white")
        plt.close()
    
    def plot_grad_flow(self, save_path, epoch_idx):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        gradient_flow_plot_save_path = os.path.join(save_path, "gradient_flow_plots")
        if not os.path.exists(gradient_flow_plot_save_path):
            os.mkdir(gradient_flow_plot_save_path)

        ave_grads = []
        max_grads= []
        layers = []
        for n, p in self.named_parameters():
            # if (p.requires_grad) and ("bias" not in n):
            if "weight" in n and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(gradient_flow_plot_save_path, "gradient_flow_ep{}".format(epoch_idx)), bbox_inches='tight', facecolor="white")
        plt.close()
    
    def visualize_activations(self, save_path, data, epoch_idx, color="C0"):
        activations = {}
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            x = embed_features(x, feature_emb_dim=self.emb_dim, value_emb_dim=0)
            activations["Embedded Feats"] = x.view(-1).numpy()

            if not self.average_pooling_flag:
                x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # x -> [num_nodes]
                cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = torch.reshape(x, (x.shape[0], (self.num_sampled_vectors + 1) * self.emb_dim))

            x = self.conv1(x, edge_index)
            activations["AMPConv Layer 1"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            x = self.norm1(x)
            activations["Layer Norm 1"] = x.view(-1).numpy()
            x = self.act1(x)
            activations["ReLU 1"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))
            # x = self.drop1(x)

            x = self.conv2(x, edge_index)
            activations["AmpConv Layer 2"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            x = self.norm2(x)
            activations["Layer Norm 2"] = x.view(-1).numpy()
            x = self.act2(x)
            activations["ReLU 2"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))
            # x = self.drop2(x)

            x = self.conv3(x, edge_index)
            activations["AmpConv Layer 3"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            x = self.norm3(x)
            activations["Layer Norm 3"] = x.view(-1).numpy()
            x = self.act3(x)
            activations["ReLU 3"] = x.view(-1).numpy()
            x = torch.reshape(x, (x.shape[0], x.shape[1] * self.emb_dim))

            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            if self.average_pooling_flag:
                x = x.mean(dim=1)  # Average pooling
            else:
                x = x[:,0]

            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            x = x.mean(dim=1)

            x = self.lin1(x)
            activations["Linear Layer 1"] = x.view(-1).numpy()
            # x = self.lin2(x)
            # activations["Linear Layer 2"] = x.view(-1).numpy()

        ## Plotting
        columns = 4
        rows = math.ceil(len(activations)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index//columns][fig_index%columns]
            sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
            key_ax.set_title(f"{key}")
            fig_index += 1
        fig.suptitle("Activation distribution", fontsize=16)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(os.path.join(save_path, "act_distrib_ep{}".format(epoch_idx)))
        plt.clf()
        plt.close()

# Create save paths
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

SAVE_PATH = os.path.join(SAVE_PATH, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
    os.system("cp experimentsd/cora_benchmark_graphsaint.py {}/".format(SAVE_PATH))
if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)
if not os.path.exists(ACTIV_PATH):
    os.mkdir(ACTIV_PATH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0].to(device)

if TRAIN_AMPCONV:
    model = AMPGCN().to(device)
else:
    model = GCN().to(device)

# batch size 1, walk length 500 => ~225 nodes
# batch size 20, walk length 100 => ~750 nodes
# batch size 10, walk length 100 => ~500 nodes
loader = GraphSAINTRandomWalkSampler(all_data, batch_size=1, walk_length=500,
                                     num_steps=1, sample_coverage=100)
# Sample one subgraph
data = iter(loader).__next__().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    total_loss = total_examples = 0
    out = model(data)
    train_loss = F.nll_loss(out, data.y, reduction='none')
    train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
    train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                data.y[data.train_mask].detach().numpy())
    
    train_loss.backward()
    if epoch % 4 == 0:
        model.plot_grad_flow(GRADS_PATH, epoch)
        model.visualize_gradients(GRADS_PATH, epoch)
        model.visualize_activations(ACTIV_PATH, data, epoch)
    optimizer.step()
    total_loss += train_loss.item() * data.num_nodes
    total_examples += data.num_nodes

    ########
    # Test #
    ########
    model.eval()
    with torch.no_grad():
        test_loss = F.nll_loss(out, data.y, reduction='none')
        test_loss = (test_loss * data.node_norm)[data.test_mask].sum()
        test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).numpy(),
                                    data.y[data.test_mask].detach().numpy())

    print("Epoch {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} "
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
    pred = model(all_data).argmax(dim=1)
    correct = (pred[all_data.test_mask] == all_data.y[all_data.test_mask]).sum()
    acc = int(correct) / int(all_data.test_mask.sum())
print(f'Final Test Accuracy: {acc:.4f}')
