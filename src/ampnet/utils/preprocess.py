import torch
import numpy as np
# from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler


def embed_features(x, feature_emb_dim=30, value_emb_dim=0):
    umap_fit = UMAP(n_neighbors=15, n_components=feature_emb_dim, min_dist=0.1, metric='euclidean')
    scaler = StandardScaler()
    emb_dim = feature_emb_dim + value_emb_dim
    num_sampled_vectors = 20

    # Feature Embedding: Perform UMAP dim reduction on transpose of nodes x features matrix
    # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
    feature_embedding = torch.from_numpy(umap_fit.fit_transform(x.numpy().transpose())) # feat embedding: [1433, feat_emb_dim=30]

    # Repeat and concatenate. Ignoring value embedding for now
    concatenated_flattened_vectors = feature_embedding.repeat(x.shape[0], 1)  # --> [1433 * num_nodes, feat_emb_dim]
    node_vectors_rolled_up = torch.reshape(concatenated_flattened_vectors,
                                      (x.shape[0],
                                       x.shape[1] * emb_dim))  # [num_nodes, 1433 * emb_dim]
    
    # First reshape into list of vectors per node
    node_vectors_unrolled = torch.reshape(node_vectors_rolled_up, (node_vectors_rolled_up.shape[0], int(node_vectors_rolled_up.shape[1] / emb_dim), emb_dim))  # [num_nodes, 1433, emb_dim]

    # Sample 20 feature vectors per node where binary value != 0
    sampled_node_vectors_unrolled = []
    for node_idx in range(x.shape[0]):
        present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
        sampled_feature_idxs = np.random.choice(present_feat_idxs, size=num_sampled_vectors, replace=True)
        sampled_vectors = node_vectors_unrolled[node_idx, sampled_feature_idxs]  # [num_sampled_vectors, emb_dim]

        # Append sampled feature vector identity indices, so model has some indication of what was sampled
        # sampled_feature_idxs_torch = torch.from_numpy(sampled_feature_idxs).unsqueeze(dim=-1)
        # sampled_vectors = torch.cat((sampled_vectors, sampled_feature_idxs_torch), dim=1)
        sampled_node_vectors_unrolled.append(sampled_vectors.unsqueeze(dim=0))
    sampled_node_vectors_unrolled = torch.cat(sampled_node_vectors_unrolled)

    # Roll vectors back up so that PyG is able to handle arrays
    node_vectors_rerolled = torch.reshape(sampled_node_vectors_unrolled, (x.shape[0], num_sampled_vectors * emb_dim))  # [num_nodes, num_sampled_vectors * emb_dim]
    
    # Z score embedding before passing on in network
    normalized_node_vectors_rolled_up_np = scaler.fit_transform(node_vectors_rerolled)
    normalized_node_vectors_rolled_up = torch.from_numpy(normalized_node_vectors_rolled_up_np).float()
    return normalized_node_vectors_rolled_up


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


Embed function with 20-feature vector sampling, PCA
def embed_features(x, feature_emb_dim, value_emb_dim):
    pca = PCA(n_components=feature_emb_dim)
    scaler = StandardScaler()
    emb_dim = feature_emb_dim + value_emb_dim
    num_sampled_vectors = 20

    # Feature Embedding: Perform PCA on transpose of nodes x features matrix
    # x is [num_nodes, 1433]. Transpose is [1433, num_nodes]
    feature_embedding = torch.from_numpy(pca.fit_transform(x.numpy().transpose())) # gene_embedding: [1433, feat_emb_dim]

    # Value Embedding: Repeat binary feature value for now
    reshaped_data = torch.reshape(x, (x.shape[0] * x.shape[1], 1))  # reshaped_data: [1433 * num_nodes, 1]
    value_embedding = reshaped_data.repeat(1, value_emb_dim)  # [1433 * num_nodes, value_emb_dim]

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
    sampled_node_vectors_unrolled = []
    for node_idx in range(x.shape[0]):
        present_feat_idxs = torch.where(x[node_idx] != 0)[0].numpy()
        if present_feat_idxs.shape[0] < num_sampled_vectors:
            sampled_feature_idxs = np.random.choice(present_feat_idxs, size=num_sampled_vectors, replace=True)
        else:
            sampled_feature_idxs = np.random.choice(present_feat_idxs, size=num_sampled_vectors, replace=False)
        
        sampled_vectors = node_vectors_unrolled[node_idx, sampled_feature_idxs]  # [num_sampled_vectors, emb_dim]
        # Append sampled feature vector identity indices, so model has some indication of what was sampled
        sampled_feature_idxs_torch = torch.from_numpy(sampled_feature_idxs).unsqueeze(dim=-1)
        sampled_vectors = torch.cat((sampled_vectors, sampled_feature_idxs_torch), dim=1)
        sampled_node_vectors_unrolled.append(sampled_vectors.unsqueeze(dim=0))
    sampled_node_vectors_unrolled = torch.cat(sampled_node_vectors_unrolled)

    # Roll vectors back up so that PyG is able to handle arrays
    node_vectors_rerolled = torch.reshape(sampled_node_vectors_unrolled, (x.shape[0], num_sampled_vectors * (emb_dim + 1)))  # [num_nodes, num_sampled_vectors * (emb_dim + 1)]
    
    # Z score embedding before passing on in network
    normalized_node_vectors_rolled_up_np = scaler.fit_transform(node_vectors_rerolled)
    normalized_node_vectors_rolled_up = torch.from_numpy(normalized_node_vectors_rolled_up_np).float()
    return normalized_node_vectors_rolled_up

"""
