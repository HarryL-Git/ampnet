import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def embed_features(x, feature_embed_dim, value_embed_dim):
    pca = PCA(n_components=feature_embed_dim)
    scaler = StandardScaler()
    emb_dim = feature_embed_dim + value_embed_dim

    # Feature Embedding: Perform PCA on transpose of nodes x features matrix
    # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
    feature_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose())) # gene_embedding: [1433, feat_emb_dim]

    # Value Embedding: Repeat binary feature value for now
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))  # reshaped_data: [1433 * num_nodes, 1]
    value_embedding = reshaped_data.repeat(1, value_embed_dim)  # [1433 * num_nodes, value_emb_dim]

    # Repeat and concatenate
    concatenated_flattened_vectors = torch.cat([
        feature_embedding.repeat(x.shape[0], 1),  # [1433 * num_nodes, feat_emb_dim]
        value_embedding], dim=1)  # --> [1433 * num_nodes, emb_dim]
    node_vectors_rolled_up = torch.reshape(concatenated_flattened_vectors,
                                      (x.shape[0],
                                       x.shape[1] * emb_dim))  # [num_nodes, 1433 * emb_dim]
    
    # First reshape into list of vectors per node
    node_vectors_unrolled = torch.reshape(node_vectors_rolled_up, (node_vectors_rolled_up.shape[0], int(node_vectors_rolled_up.shape[1] / emb_dim), emb_dim))  # [num_nodes, 1433, emb_dim]

    # Sample 20 feature vectors per node where binary value != 0
    sampled_node_vectors_unrolled = torch.where(node_vectors_unrolled)
    for node_idx in range(x.shape[0]):
        present_feat_idxs = torch.where(x[node_idx] != 0)[0]
        # sampled_features = 
    
    # Z score embedding before passing on in network
    normalized_node_vectors_rolled_up = scaler.fit_transform(node_vectors_rolled_up)
    node_vectors_rolled_up = torch.from_numpy(normalized_node_vectors_rolled_up).float()
    return node_vectors_rolled_up


"""
Untouched embed function
def embed_features(x, feature_embed_dim, value_embed_dim):
    pca = PCA(n_components=feature_embed_dim)
    scaler = StandardScaler()

    # Feature Embedding: Perform PCA on transpose of nodes x features matrix
    # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
    gene_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose())) # gene_embedding: [1433, feat_emb_dim]
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))  # reshaped_data: [1433 * num_nodes, 1]
    genes_with_embedding = torch.cat([
        gene_embedding.repeat(x.shape[0], 1),  # [1433 * num_nodes, feat_emb_dim]
        reshaped_data.repeat(1, value_embed_dim)], dim=1)  # [1433 * num_nodes, value_emb_dim]
    embedded_per_spot = torch.reshape(genes_with_embedding,
                                      (x.shape[0],
                                       x.shape[1] * (feature_embed_dim + value_embed_dim)))  # [num_nodes, 1433 * (feat+emb_dim)]
    
    # Z score embedding before passing on in network
    normalized_embedded_per_spot = scaler.fit_transform(embedded_per_spot)
    embedded_per_spot = torch.from_numpy(normalized_embedded_per_spot).float()
    return embedded_per_spot


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
