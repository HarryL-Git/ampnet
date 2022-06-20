import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid


def plot_PCA_2D(embedding_arr, plot_name, plot_title):
    plt.title(plot_title)
    sns.scatterplot(x=embedding_arr[:,0], y=embedding_arr[:,1])
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.savefig("./{}.png".format(plot_name), facecolor="white", bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_PCA_cumulative_expl_var_ratio(data):
    plt.title("Cumulative PCA Explained Variance Ratio")
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("./cumulative_pca_expl_var_ratio.png", facecolor="white", bbox_inches="tight")
    # plt.show()
    plt.close()

# Prepare data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)
x, edge_index = data.x, data.edge_index
feature_embed_dim = 2
# x = embed_features(x, feature_embed_dim=4, value_embed_dim=1)

# Plot figure for cumulative variance explained ratio figure
plot_PCA_cumulative_expl_var_ratio(x.numpy().transpose())

# Plot 2D component plot of prior embedding to visualize if features are being differentiated
# pca = PCA(n_components=feature_embed_dim)
# prior_embedding_np = pca.fit_transform(x.numpy().transpose())

# print(prior_embedding_np.shape)
# plot_PCA_2D(prior_embedding_np, "Cora_PCA_embedding_2D_plot", "Cora PCA Embedding 2D Plot")

# # Normalize PCA components using Z-score
# scaler = StandardScaler()
# normalized_prior_embedding_np = scaler.fit_transform(prior_embedding_np)
# plot_PCA_2D(normalized_prior_embedding_np, "Cora_normalized_PCA_emb_2D_plot", "Cora Normalized PCA Embedding 2D Plot")
