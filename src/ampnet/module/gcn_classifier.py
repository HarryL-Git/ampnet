import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils.dropout import dropout_adj


class GCN(torch.nn.Module):
    def __init__(self, 
                device="cpu", 
                num_node_features=1433, 
                hidden_dim=16,
                num_sampled_vectors=40,
                output_dim=7, 
                softmax_out=True, 
                feat_emb_dim=99, 
                val_emb_dim=1,
                downsample_feature_vectors=True,
                dropout_rate=0.1,
                dropout_adj_rate=0.1):
        super().__init__()
        # print("Initializing GCN network...")
        self.device = device
        self.emb_dim = feat_emb_dim + val_emb_dim
        self.num_sampled_vectors = num_sampled_vectors
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.softmax_out = softmax_out
        self.feat_emb_dim = feat_emb_dim
        self.val_emb_dim = val_emb_dim
        self.downsample_feature_vectors = downsample_feature_vectors
        self.dropout_rate = dropout_rate
        self.dropout_adj_rate = dropout_adj_rate

        channels = num_node_features * self.emb_dim
        # self.mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))

        self.feature_embedding_table = nn.Embedding(
            num_embeddings=num_node_features,
            embedding_dim=feat_emb_dim
        )

        self.conv1 = GCNConv(channels, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.act_out = nn.Sigmoid()
    
        # self.initialize_weights()
    
    # def initialize_weights(self):
    #     # nn.init.normal_(self.cls_token, std=.02)
    #     nn.init.normal_(self.mask_token, std=.02)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # x is [2708, 1433]
        edge_index = dropout_adj(edge_index=edge_index, p=self.dropout_adj_rate, training=self.training)[0]
        # x = self.sample_feats_and_mask(x.to("cpu"))
        x = self.normalize_features_and_add_feature_embedding(x.to("cpu"))
        # x = self.normalize_features(x.to("cpu"))
        x = x.to(self.device)

        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.conv2(x, edge_index)

        if self.softmax_out:
            return F.log_softmax(x, dim=1)
        else:
            return self.act_out(x)
    
    def normalize_features(self, x):
        # Use StandardScaler (z-scoring) to normalize two XOR features, transform to torch tensor
        scaler = StandardScaler()
        x_ = scaler.fit_transform(x.numpy())
        x_ = torch.from_numpy(x_)
        x_ = x_.requires_grad_(True)
        return x_
    
    def normalize_features_and_add_feature_embedding(self, x):
        # Use StandardScaler (z-scoring) to normalize two XOR features, transform to torch tensor
        scaler = StandardScaler()
        x_ = scaler.fit_transform(x.numpy())
        x_ = torch.from_numpy(x_)
        x_ = x_.requires_grad_(True)

        # Add feature embedding from table to each feature in each node
        node_vectors_unrolled = []
        for node_idx in range(x.shape[0]):
            x_embed = torch.cat((self.feature_embedding_table.weight, x_[node_idx].unsqueeze(-1)), dim=1)
            node_vectors_unrolled.append(x_embed.unsqueeze(dim=0))
        
        # Concatenate and reshape into flatten node flattened embeddings
        node_vectors_unrolled = torch.cat(node_vectors_unrolled)
        node_vectors_rerolled = torch.reshape(node_vectors_unrolled, (x.shape[0], self.num_sampled_vectors * self.emb_dim))

        node_vectors_rerolled = node_vectors_rerolled.requires_grad_(True)
        return node_vectors_rerolled
    
    def sample_feats_and_mask(self, x):
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
        
        if self.downsample_feature_vectors:
            # First reshape into list of vectors per node
            node_vectors_unrolled = torch.reshape(node_vectors_rolled_up, (node_vectors_rolled_up.shape[0], int(node_vectors_rolled_up.shape[1] / self.emb_dim), self.emb_dim))  # [num_nodes, 1433, emb_dim]

            # Sample 20 feature vectors, skewing to balance 1s and 0s. Mask out all unsampled vectors with mask tokens
            for node_idx in range(x.shape[0]):
                present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
                unpresent_indices_len = self.num_node_features - len(present_feat_idxs)

                sampling_probs = [0.5 / len(present_feat_idxs) if idx in present_feat_idxs else 0.5 / unpresent_indices_len for idx in range(self.num_node_features)]

                feat_indices = list(range(self.num_node_features))
                sampled_feature_idxs = np.random.choice(feat_indices, size=self.num_sampled_vectors, replace=False, p=sampling_probs)

                # Mask out all unsampled feature vectors with mask token
                for feat_idx in range(node_vectors_unrolled.shape[1]):
                    if feat_idx not in sampled_feature_idxs:
                        node_vectors_unrolled[node_idx, feat_idx] = self.mask_token

            # Roll vectors back up so that PyG is able to handle arrays
            node_vectors_rerolled = torch.reshape(node_vectors_unrolled, (x.shape[0], node_vectors_unrolled.shape[1] * self.emb_dim))  # [num_nodes, num_feats * emb_dim]
        else:
            node_vectors_rerolled = node_vectors_rolled_up

        node_vectors_rolled_up = node_vectors_rerolled.requires_grad_(True)
        
        return node_vectors_rolled_up

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
            key_ax = ax[fig_index%columns]
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
            x, edge_index = data.x, data.edge_index  # x is [2708, 1433]
            # edge_index = dropout_adj(edge_index=edge_index, p=self.dropout_adj_rate, training=self.training)[0]
            # x = self.sample_feats_and_mask(x.to("cpu"))
            x = self.normalize_features_and_add_feature_embedding(x.to("cpu"))
            # x = self.normalize_features(x.to("cpu"))
            x = x.to(self.device)
            # activations["Embedded Feats"] = x.view(-1).cpu().numpy()

            x = self.conv1(x, edge_index)
            activations["GCN Layer 1"] = x.view(-1).cpu().numpy()
            # x = self.norm1(x)
            # activations["Norm Layer 1"] = x.view(-1).numpy()
            x = self.act1(x)
            activations["ReLU 1"] = x.view(-1).cpu().numpy()
            x = self.drop1(x)
            x = self.conv2(x, edge_index)
            activations["GCN Layer 2"] = x.view(-1).cpu().numpy()

        ## Plotting
        columns = 2
        rows = math.ceil(len(activations)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index//columns][fig_index%columns]
            sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
            key_ax.set_title(f"Layer {key}")
            fig_index += 1
        fig.suptitle("Activation distribution", fontsize=16)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(os.path.join(save_path, "act_distrib_ep{}_iter{}".format(epoch_idx, iter)))
        plt.clf()
        plt.close()
