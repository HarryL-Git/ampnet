import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid


def plot_PCA_2D(embedding_arr):
    plt.title("Cora PCA Embedding 2D Plot")
    sns.scatterplot(x=embedding_arr[:,0], y=embedding_arr[:,1])
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.savefig("./Cora_PCA_embedding_2D_plot", facecolor="white", bbox_inches="tight")
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)

x, edge_index = data.x, data.edge_index
feature_embed_dim = 2
# x = embed_features(x, feature_embed_dim=4, value_embed_dim=1)

pca = PCA(n_components=feature_embed_dim)
prior_embedding_np = pca.fit_transform(x.numpy().transpose())

print(prior_embedding_np.shape)

plot_PCA_2D(prior_embedding_np)

