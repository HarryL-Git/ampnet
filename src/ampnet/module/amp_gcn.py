import os
import math

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
from torch_geometric.utils.dropout import dropout_adj
from src.ampnet.conv.amp_conv import AMPConv
from src.ampnet.utils.utils import *


class AMPGCN(torch.nn.Module):
    def __init__(self, 
                device="cpu", 
                embedding_dim=100, 
                num_heads=2,
                num_node_features=1433, 
                num_sampled_vectors=40,
                output_dim=7, 
                softmax_out=True, 
                feat_emb_dim=99, 
                val_emb_dim=1,
                downsample_feature_vectors=True,
                average_pooling_flag=True,
                dropout_rate=0.1,
                dropout_adj_rate=0.1,
                feature_repeats=5):
        super().__init__()
        assert embedding_dim == feat_emb_dim + val_emb_dim, "Feature and value dimensions do not add up to total embedding dimension"

        self.device = device
        self.conv1_embedding = None
        self.conv2_embedding = None
        self.emb_dim = embedding_dim
        self.num_sampled_vectors = num_sampled_vectors
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.softmax_out = softmax_out
        self.feat_emb_dim = feat_emb_dim
        self.val_emb_dim = val_emb_dim
        self.downsampling_vectors = downsample_feature_vectors
        self.average_pooling_flag = average_pooling_flag
        self.dropout_rate = dropout_rate
        self.dropout_adj_rate = dropout_adj_rate
        self.feature_repeats = feature_repeats

        # num_flattened_features = self.num_sampled_vectors * self.emb_dim if downsample_feature_vectors else dataset.num_node_features * self.emb_dim
        self.feature_embedding_table = nn.Embedding(
            num_embeddings=num_node_features,  #  XOR: // feature_repeats
            embedding_dim=feat_emb_dim
        )
        # self.mask_token = nn.Parameter(torch.zeros(1, embedding_dim))

        if not average_pooling_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            nn.init.normal_(self.cls_token, std=.02)

        # For Block 1
        # self.layer_norm1 = nn.LayerNorm(
        #     self.emb_dim,
        #     elementwise_affine=False  # Masked AutoEncoder has this to True, Pytorch default
        # )

        # self.layer_norm2 = nn.LayerNorm(
        #     self.emb_dim,
        #     elementwise_affine=False
        # )

        self.conv1 = AMPConv(
            embed_dim=embedding_dim,
            num_heads=num_heads
        )
        # self.post_conv_linear1 = nn.Linear(
        #     in_features=self.emb_dim,
        #     out_features=self.emb_dim
        # )

        self.drop1 = nn.Dropout(p=dropout_rate)

        # For Block 2
        # self.layer_norm3 = nn.LayerNorm(
        #     self.emb_dim,
        #     elementwise_affine=False
        # )

        # self.layer_norm4 = nn.LayerNorm(
        #     self.emb_dim,
        #     elementwise_affine=False
        # )

        # self.conv2 = AMPConv(
        #     embed_dim=embedding_dim,
        #     num_heads=num_heads
        # )

        # self.post_conv_linear2 = nn.Linear(
        #     in_features=self.emb_dim,
        #     out_features=self.emb_dim
        # )

        # self.drop2 = nn.Dropout(p=dropout_rate)

        self.final_linear_out = nn.Linear(
            in_features=self.emb_dim,
            out_features=output_dim
        )

        self.drop3 = nn.Dropout(p=dropout_rate)
        self.act_out = nn.Sigmoid()

    def normalize_features_and_add_feature_table_embedding(self, x):
        # Use StandardScaler (z-scoring) to normalize two XOR features, transform to torch tensor
        scaler = StandardScaler()
        x_ = scaler.fit_transform(x.cpu().numpy())
        x_ = torch.from_numpy(x_)
        x_ = x_.to(self.device).requires_grad_(True)

        if self.downsampling_vectors:
            # x shape: [num_nodes, 1433]
            sampled_node_vectors_unrolled = []
            sampled_indices = []

            for node_idx in range(x.shape[0]):
                present_feat_idxs = torch.where(x[node_idx] != 0)[0]
                unpresent_indices_len = self.num_node_features - len(present_feat_idxs)

                sampling_probs = [0.5 / len(present_feat_idxs) if idx in present_feat_idxs else 0.5 / unpresent_indices_len for idx in range(self.num_node_features)]
                feat_indices = list(range(self.num_node_features))
                sampled_feature_idxs = np.random.choice(feat_indices, size=self.num_sampled_vectors, replace=False,
                                                        p=sampling_probs)
                sampled_vectors = self.feature_embedding_table.weight[sampled_feature_idxs]
                sampled_vectors = torch.cat((sampled_vectors, x_[node_idx, sampled_feature_idxs].unsqueeze(-1)), dim=1)

                sampled_node_vectors_unrolled.append(sampled_vectors.unsqueeze(dim=0))
                sampled_indices.append(np.expand_dims(sampled_feature_idxs, axis=0))
            sampled_node_vectors_unrolled = torch.cat(sampled_node_vectors_unrolled)
            sampled_indices = np.concatenate(sampled_indices)
            node_vectors_rerolled = torch.reshape(sampled_node_vectors_unrolled,
                                                  (x_.shape[0], self.num_sampled_vectors * self.emb_dim))

            # Uncomment for XOR synthetic benchmark implementation
            # sampled_node_vectors_unrolled = []
            # for node_idx in range(x_.shape[0]):
            #     present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
            #     feat_idxs = list(range(self.num_node_features))
            #     sampled_feature_idxs = np.random.choice(feat_idxs, size=self.num_sampled_vectors, replace=True)
            #     # sampled_vectors = self.feature_embedding_table.weight[feat_idxs]  # [num_sampled_vectors, emb_dim]
            #     sampled_vectors = torch.tile(self.feature_embedding_table.weight, [self.feature_repeats, 1])[sampled_feature_idxs]  # [num_sampled_vectors, emb_dim]
            #     sampled_vectors = torch.cat((sampled_vectors, x_[node_idx, sampled_feature_idxs].unsqueeze(-1)), dim=1)
            #     sampled_node_vectors_unrolled.append(sampled_vectors.unsqueeze(dim=0))
            #
            # sampled_node_vectors_unrolled = torch.cat(sampled_node_vectors_unrolled)
            # node_vectors_rerolled = torch.reshape(sampled_node_vectors_unrolled, (x_.shape[0], self.num_sampled_vectors * self.emb_dim))
        else:
            # Add feature embedding from table to each feature in each node
            node_vectors_unrolled = []
            for node_idx in range(x_.shape[0]):
                x_embed = torch.cat((
                    torch.tile(self.feature_embedding_table.weight, [self.feature_repeats, 1]), 
                    x_[node_idx].unsqueeze(-1)), dim=1)
                node_vectors_unrolled.append(x_embed.unsqueeze(dim=0))
            
            # Concatenate and reshape into flatten node flattened embeddings
            node_vectors_unrolled = torch.cat(node_vectors_unrolled)
            node_vectors_rerolled = torch.reshape(node_vectors_unrolled, (x_.shape[0], self.num_sampled_vectors * self.emb_dim))

        node_vectors_rerolled = node_vectors_rerolled.requires_grad_(True)
        return node_vectors_rerolled

    def normalize_features_and_add_pca_feature_embedding(self, x):
        assert self.emb_dim == self.feat_emb_dim + self.val_emb_dim, "feat and val emb dim must match self.emb_dim"
        pca = PCA(n_components=self.feat_emb_dim)
        scaler = StandardScaler()

        # Use StandardScaler (z-scoring) to normalize two XOR features, transform to torch tensor
        scaler = StandardScaler()
        x_ = scaler.fit_transform(x.numpy())
        x_ = torch.from_numpy(x_)

        # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
        feature_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose()))  # feat embedding: [1433, feat_emb_dim]
        reshaped_data = torch.reshape(x_, (x_.shape[0] * x_.shape[1], 1))  # reshaped_data: [1433 * num_nodes, 1]

        # Repeat and concatenate
        concatenated_flattened_vectors = torch.cat([
            feature_embedding.repeat(x.shape[0], 1),  # [1433 * num_nodes, feat_emb_dim]
            reshaped_data.repeat(1, self.val_emb_dim)], dim=1)  # [1433 * num_nodes, self.val_emb_dim]

        node_vectors_rolled_up = torch.reshape(concatenated_flattened_vectors,
                                        (x.shape[0],
                                        x.shape[1] * self.emb_dim))  # [num_nodes, 1433 * emb_dim]
        
        if self.downsampling_vectors:
            # First reshape into list of vectors per node
            node_vectors_unrolled = torch.reshape(node_vectors_rolled_up, (node_vectors_rolled_up.shape[0], int(node_vectors_rolled_up.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]

            # Sample 20 feature vectors per node where binary value != 0
            sampled_node_vectors_unrolled = []
            sampled_indices = []
            for node_idx in range(x.shape[0]):
                present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
                unpresent_indices_len = self.num_node_features - len(present_feat_idxs)

                sampling_probs = [0.5 / len(present_feat_idxs) if idx in present_feat_idxs else 0.5 / unpresent_indices_len for idx in range(self.num_node_features)]

                feat_indices = list(range(self.num_node_features))
                sampled_feature_idxs = np.random.choice(feat_indices, size=self.num_sampled_vectors, replace=False, p=sampling_probs)
                sampled_vectors = node_vectors_unrolled[node_idx, sampled_feature_idxs]  # [num_sampled_vectors, emb_dim]

                sampled_node_vectors_unrolled.append(sampled_vectors.unsqueeze(dim=0))
                sampled_indices.append(np.expand_dims(sampled_feature_idxs, axis=0))
            sampled_node_vectors_unrolled = torch.cat(sampled_node_vectors_unrolled)
            sampled_indices = np.concatenate(sampled_indices)

            # Roll vectors back up so that PyG is able to handle arrays
            node_vectors_rerolled = torch.reshape(sampled_node_vectors_unrolled, (x.shape[0], self.num_sampled_vectors * self.emb_dim))  # [num_nodes, num_sampled_vectors * emb_dim]
        else:
            node_vectors_rerolled = node_vectors_rolled_up
            sampled_indices = None

        node_vectors_rolled_up = node_vectors_rerolled.requires_grad_(True)
        return node_vectors_rolled_up, sampled_indices

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        edge_index = dropout_adj(edge_index=edge_index, p=self.dropout_adj_rate, training=self.training)[0]
        # x, sampled_indices = self.normalize_features_and_add_pca_feature_embedding(x.to(self.device))
        x = self.normalize_features_and_add_feature_table_embedding(x)
        x = x.to(self.device)

        x = self.drop1(x)
        x = self.conv1(x, edge_index)
        self.conv1_embedding = x
        # Layer normalization
        # x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
        # x = self.layer_norm1(x)
        # x = torch.reshape(x, (x.shape[0], self.num_sampled_vectors * self.emb_dim))

        # x = F.elu(x)
        x = F.relu(x)

        # x = self.drop2(x)
        # x = self.conv2(x, edge_index)
        # self.conv2_embedding = x
        # x = F.elu(x)

        x = self.drop3(x)
        # Reshape node features into unrolled list of feature vectors, perform average pooling
        x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]
        if self.average_pooling_flag:
            x = x.mean(dim=1)  # Average pooling
        else:
            x = x[:, 0]

        x = self.final_linear_out(x)
        if self.softmax_out:
            return F.log_softmax(x, dim=1)
        else:
            return self.act_out(x)
    
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
            key_ax = ax[fig_index % columns]
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
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
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
            x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)  # x: [num_nodes, 1433]
            edge_index = dropout_adj(edge_index=edge_index, p=self.dropout_adj_rate, training=self.training)[0]
            # x, sampled_indices = self.normalize_features_and_add_pca_feature_embedding(x.to(self.device))
            x = self.normalize_features_and_add_feature_table_embedding(x)
            x = x.to(self.device)

            x = self.drop1(x)
            x = self.conv1(x, edge_index)
            activations["AmpConv 1"] = x.view(-1).cpu().numpy()
            self.conv1_embedding = x
            # Layer normalization
            # x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
            # x = self.layer_norm1(x)
            # activations["LayerNorm 1"] = x.view(-1).cpu().numpy()
            # x = torch.reshape(x, (x.shape[0], self.num_sampled_vectors * self.emb_dim))

            # x = F.elu(x)
            x = F.relu(x)
            activations["ReLU 1"] = x.view(-1).cpu().numpy()

            # x = self.drop2(x)
            # x = self.conv2(x, edge_index)
            # activations["AmpConv 2"] = x.view(-1).cpu().numpy()
            # self.conv2_embedding = x
            # x = F.elu(x)
            # activations["ELU 1"] = x.view(-1).cpu().numpy()

            x = self.drop3(x)
            # Reshape node features into unrolled list of feature vectors, perform average pooling
            x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]
            if self.average_pooling_flag:
                x = x.mean(dim=1)  # Average pooling
                activations["Average Pooling"] = x.view(-1).cpu().numpy()
            else:
                x = x[:, 0]
                activations["Class Token"] = x.view(-1).cpu().numpy()

            x = self.final_linear_out(x)
            activations["Linear Out"] = x.view(-1).cpu().numpy()

        # Plotting
        columns = 2
        rows = math.ceil(len(activations)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index//columns][fig_index % columns]
            sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
            key_ax.set_title(f"{key}")
            fig_index += 1
        fig.suptitle("Activation distribution", fontsize=16)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(os.path.join(save_path, "act_distrib_ep{}_iter{}".format(epoch_idx, iter)))
        plt.clf()
        plt.close()


"""
Transformer block forward function
#---------- Block 1 ----------->
x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm1(x_)  # Pre-LN Transformer
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x_ = self.conv1(x_, edge_index)
self.conv1_embedding = x_
x = self.drop1(x)
x = x + x_  # First skip connection

x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm2(x_)
x_ = self.post_conv_linear1(x_)
x_ = F.elu(x_)
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x = x + x_  # Second skip connection

# ---------- Block 2 ----------->
x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm3(x_)  # Pre-LN Transformer
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x_ = self.conv2(x_, edge_index)
self.conv2_embedding = x_
x = self.drop2(x)
x = x + x_  # First skip connection

x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm4(x_)
x_ = self.post_conv_linear2(x_)
x_ = F.elu(x_)
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x = x + x_  # Second skip connection

# Reshape node features into unrolled list of feature vectors, perform average pooling
x = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]
if self.average_pooling_flag:
    x = x.mean(dim=1)  # Average pooling
else:
    x = x[:, 0]

x = self.final_linear_out(x)
return F.log_softmax(x, dim=1)


Corresponding visualize_activations forward pass
x, edge_index = data.x, data.edge_index  # x: [num_nodes, 1433]
x, sampled_indices = self.embed_features_and_downsample(x.to(self.device))
x = x.to(self.device)
# x becomes [num_nodes, num_feats * emb_dim], sampled_indices are [num_nodes, 20]

# ---------- Block 1 ----------->
x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm1(x_)  # Pre-LN Transformer
activations["Norm 1"] = x_.view(-1).cpu().numpy()
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x_ = self.conv1(x_, edge_index)
activations["AmpConv 1"] = x_.view(-1).cpu().numpy()
self.conv1_embedding = x_
x = self.drop1(x)
x = x + x_  # First skip connection

x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm2(x_)
activations["Norm 2"] = x_.view(-1).cpu().numpy()
x_ = self.post_conv_linear1(x_)
activations["Post Conv Linear 1"] = x_.view(-1).cpu().numpy()
x_ = F.elu(x_)
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x = x + x_  # Second skip connection

# ---------- Block 2 ----------->
x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm3(x_)  # Pre-LN Transformer
activations["Norm 3"] = x_.view(-1).cpu().numpy()
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x_ = self.conv2(x_, edge_index)
activations["AmpConv 2"] = x_.view(-1).cpu().numpy()
self.conv2_embedding = x_
x = self.drop2(x)
x = x + x_  # First skip connection

x_ = torch.reshape(x, (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))
x_ = self.layer_norm4(x_)
activations["Norm 4"] = x_.view(-1).cpu().numpy()
x_ = self.post_conv_linear2(x_)
activations["Post Conv Linear 2"] = x_.view(-1).cpu().numpy()
x_ = F.elu(x_)
x_ = torch.reshape(x_, (x.shape[0], self.num_sampled_vectors * self.emb_dim))
x = x + x_  # Second skip connection

# Reshape node features into unrolled list of feature vectors, perform average pooling
x = torch.reshape(x,
                    (x.shape[0], int(x.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]
if self.average_pooling_flag:
    x = x.mean(dim=1)  # Average pooling
else:
    x = x[:, 0]

x = self.final_linear_out(x)
activations["Linear Layer 1"] = x_.view(-1).cpu().numpy()

"""
