import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def embed_features(x, feature_embed_dim, value_embed_dim):
    pca = PCA(n_components=feature_embed_dim)
    scaler = StandardScaler()

    gene_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose()))
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))
    genes_with_embedding = torch.cat([
        gene_embedding.repeat(x.shape[0], 1),
        reshaped_data.repeat(1, value_embed_dim)], dim=1)
    embedded_per_spot = torch.reshape(genes_with_embedding,
                                      (x.shape[0],
                                       x.shape[1] * (feature_embed_dim + value_embed_dim)))
    
    # Z score embedding before passing on in network
    normalized_embedded_per_spot = scaler.fit_transform(embedded_per_spot)
    embedded_per_spot = torch.from_numpy(normalized_embedded_per_spot).float()
    return embedded_per_spot


"""
# embed_features function that z-scores pca component embeddings before adding in feature values.
# This does not train well, 2-layer GCN cannot learn on this data.
def embed_features(x, feature_embed_dim, value_embed_dim):
    pca = PCA(n_components=feature_embed_dim)
    scaler = StandardScaler()

    gene_embedding_np = pca.fit_transform(x.numpy().transpose())
    gene_embedding_np_normalized = scaler.fit_transform(gene_embedding_np)
    gene_embedding = torch.from_numpy(gene_embedding_np_normalized)
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))
    genes_with_embedding = torch.cat([
        gene_embedding.repeat(x.shape[0], 1),
        reshaped_data.repeat(1, value_embed_dim)], dim=1)
    embedded_per_spot = torch.reshape(genes_with_embedding,
                                      (x.shape[0],
                                       x.shape[1] * (feature_embed_dim + value_embed_dim)))
    
    return embedded_per_spot
"""
