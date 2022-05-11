import torch
from sklearn.decomposition import PCA


def embed_features(x, feature_embed_dim, value_embed_dim):
    pca = PCA(n_components=feature_embed_dim)
    gene_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose()))
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))
    genes_with_embedding = torch.cat([
        gene_embedding.repeat(x.shape[0], 1),
        reshaped_data.repeat(1, value_embed_dim)], dim=1)
    embedded_per_spot = torch.reshape(genes_with_embedding,
                                      (x.shape[0],
                                       x.shape[1] * (feature_embed_dim + value_embed_dim)))
    return embedded_per_spot
