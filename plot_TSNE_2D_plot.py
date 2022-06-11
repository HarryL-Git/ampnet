import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid


def plot_TSNE_2D(embedding_arr, plot_name, plot_title):
    plt.title(plot_title)
    sns.scatterplot(x=embedding_arr[:,0], y=embedding_arr[:,1])
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    plt.savefig("./{}.png".format(plot_name), facecolor="white", bbox_inches="tight")
    # plt.show()
    plt.close()


# Prepare data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)
x, edge_index = data.x, data.edge_index
feature_embed_dim = 2
# x = embed_features(x, feature_embed_dim=4, value_embed_dim=1)

# Plot 2D component plot of prior embedding to visualize if features are being differentiated
tsne = TSNE(n_components=feature_embed_dim, perplexity=10, learning_rate=10)
prior_embedding_np = tsne.fit_transform(x.numpy().transpose())

print(prior_embedding_np.shape)
plot_TSNE_2D(prior_embedding_np, "Cora_tSNE_emb_2D_plot", "Cora tSNE Embedding 2D Plot")

# Normalize TSNE components using Z-score
scaler = StandardScaler()
normalized_prior_embedding_np = scaler.fit_transform(prior_embedding_np)
plot_TSNE_2D(normalized_prior_embedding_np, "Cora_normalized_tSNE_emb_2D_plot", "Cora Normalized tSNE Embedding 2D Plot")
