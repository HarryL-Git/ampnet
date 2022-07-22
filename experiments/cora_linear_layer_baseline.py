import os
import math
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils.dropout import dropout_adj
from src.ampnet.utils.utils import *


# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0].to(device)


# Create save paths
save_path = "./experiments/runs_linear_layer"
if not os.path.exists(save_path):
    os.mkdir(save_path)

SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
    os.system("cp ./experiments/cora_total_pooling.py {}/".format(SAVE_PATH))
if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)
if not os.path.exists(ACTIV_PATH):
    os.mkdir(ACTIV_PATH)


class LinearLayerModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        print("Initializing Linear Layer network...")
        self.device = device
        self.emb_dim = 100
        self.num_sampled_vectors = 40
        self.mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))

        self.drop1 = nn.Dropout(p=0.2)

        self.lin1 = nn.Linear(in_features=dataset.num_node_features * self.emb_dim, out_features=dataset.num_classes)
    
        self.initialize_weights()
    
    def initialize_weights(self):
        # nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [2708, 1433]
        edge_index = dropout_adj(edge_index=edge_index, p=0.2, training=self.training)[0]
        # x = self.embed_features(x, feature_embed_dim=5, value_embed_dim=1)  # x becomes [2708, 8598]
        x = self.sample_feats_and_mask(x.to("cpu"))
        x = x.to(self.device)

        s = self.drop1(x)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)
    
    def sample_feats_and_mask(self, x, feature_emb_dim=99, value_emb_dim=1):
        assert self.emb_dim == feature_emb_dim + value_emb_dim, "feat and val emb dim must match self.emb_dim"
        pca = PCA(n_components=feature_emb_dim)
        scaler = StandardScaler()

        # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
        feature_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose()))  # feat embedding: [1433, feat_emb_dim]
        reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))  # reshaped_data: [1433 * num_nodes, 1]

        # Repeat and concatenate
        concatenated_flattened_vectors = torch.cat([
            feature_embedding.repeat(x.shape[0], 1),  # [1433 * num_nodes, feat_emb_dim]
            reshaped_data.repeat(1, value_emb_dim)], dim=1)  # [1433 * num_nodes, value_emb_dim]

        node_vectors_rolled_up = torch.reshape(concatenated_flattened_vectors,
                                        (x.shape[0],
                                        x.shape[1] * self.emb_dim))  # [num_nodes, 1433 * emb_dim]
        
        # First reshape into list of vectors per node
        node_vectors_unrolled = torch.reshape(node_vectors_rolled_up, (node_vectors_rolled_up.shape[0], int(node_vectors_rolled_up.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]

        # Sample 20 feature vectors, skewing to balance 1s and 0s. Mask out all unsampled vectors with mask tokens
        for node_idx in range(x.shape[0]):
            present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
            unpresent_indices_len = dataset.num_node_features - len(present_feat_idxs)

            sampling_probs = [0.5 / len(present_feat_idxs) if idx in present_feat_idxs else 0.5 / unpresent_indices_len for idx in range(dataset.num_node_features)]

            feat_indices = list(range(dataset.num_node_features))
            sampled_feature_idxs = np.random.choice(feat_indices, size=self.num_sampled_vectors, replace=False, p=sampling_probs)

            # Mask out all unsampled feature vectors with mask token
            for feat_idx in range(node_vectors_unrolled.shape[1]):
                if feat_idx not in sampled_feature_idxs:
                    node_vectors_unrolled[node_idx, feat_idx] = self.mask_token

        # Roll vectors back up so that PyG is able to handle arrays
        node_vectors_rerolled = torch.reshape(node_vectors_unrolled, (x.shape[0], node_vectors_unrolled.shape[1] * self.emb_dim))  # [num_nodes, num_feats * emb_dim]
        
        # Z score embedding before passing on in network
        normalized_node_vectors_rolled_up_np = scaler.fit_transform(node_vectors_rerolled.detach().numpy())
        normalized_node_vectors_rolled_up = torch.tensor(normalized_node_vectors_rolled_up_np, requires_grad=True).float()  # torch.from_numpy(normalized_node_vectors_rolled_up_np).float()
        
        return normalized_node_vectors_rolled_up

    def visualize_gradients(self, save_path, epoch_idx, iter, color="C0"):
        """
        Visualization code partly taken from the notebook tutorial at
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
        """
        grads = {name: params.grad.data.view(-1).cpu() for name, params in list(self.named_parameters()) if "weight" in name and params.grad is not None}  # Only seeing gradients for leaf nodes, non-leaves grad is None

        gradient_distrib_save_path = os.path.join(save_path, "gradient_distrib_plots")
        if not os.path.exists(gradient_distrib_save_path):
            os.mkdir(gradient_distrib_save_path)
        
        columns = len(grads)
        fig, ax = plt.subplots(1, columns, figsize=(columns*4, 4))
        fig_index = 0
        for key in grads:
            key_ax = ax
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
        plt.savefig(os.path.join(gradient_distrib_save_path, "gradient_distrib_epoch{}_itr{}".format(epoch_idx, iter)), bbox_inches='tight', facecolor="white")
        plt.close()
    
    def plot_grad_flow(self, save_path, epoch_idx, iter):
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
        max_grads = []
        layers = []
        for n, p in self.named_parameters():
            # if (p.requires_grad) and ("bias" not in n):
            if "weight" in n and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
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
        plt.savefig(os.path.join(gradient_flow_plot_save_path, "gradient_flow_ep{}_itr{}".format(epoch_idx, iter)), bbox_inches='tight', facecolor="white")
        plt.close()
    
    def visualize_activations(self, save_path, data, epoch_idx, iter, color="C0"):
        activations = {}
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            # x = self.embed_features(x, feature_embed_dim=5, value_embed_dim=1)
            x = self.sample_feats_and_mask(x.to("cpu"))
            x = x.to(self.device)
            activations["Embedded Feats"] = x.view(-1).cpu().numpy()

            x = self.lin1(x)
            activations["Linear Layer"] = x.view(-1).cpu().numpy()

        ## Plotting
        columns = 1
        rows = math.ceil(len(activations)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index//columns]
            sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
            key_ax.set_title(f"Layer {key}")
            fig_index += 1
        fig.suptitle("Activation distribution", fontsize=16)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(os.path.join(save_path, "act_distrib_ep{}_iter{}".format(epoch_idx, iter)))
        plt.clf()
        plt.close()


model = LinearLayerModel().to(device)

loader = GraphSAINTRandomWalkSampler(all_data, batch_size=10, walk_length=150,
                                     num_steps=10, sample_coverage=100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 15
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
        train_loss = F.nll_loss(out, data.y, reduction='none')
        train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
        train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                    data.y[data.train_mask].detach().numpy())
        
        train_loss.backward()
        if idx % 4 == 0:
            model.plot_grad_flow(GRADS_PATH, epoch, idx)
            model.visualize_gradients(GRADS_PATH, epoch, idx)
            model.visualize_activations(ACTIV_PATH, data, epoch, idx)
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

        print("Epoch {:05d} Partition {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} "
                .format(epoch, idx, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_accuracy)
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_accuracy)


print("Training took {} minutes.".format((time.time() - start_time) / 60.))
plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="LinearLayer")
plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="LinearLayer")

model.eval()
with torch.no_grad():
    pred = model(all_data).argmax(dim=1)
    correct = (pred[all_data.test_mask] == all_data.y[all_data.test_mask]).sum()
    acc = int(correct) / int(all_data.test_mask.sum())
print(f'Final Test Accuracy: {acc:.4f}')
