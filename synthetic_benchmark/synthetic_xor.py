import os
import random
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

"""
XOR Rules
Feature1 - Feature2 - XOR
0 - 0 - 0
0 - 1 - 1
1 - 0 - 1
1 - 1 - 0

Notes:
- Not linearly separable. A linear layer itself cannot learn it, but a linear layer with a nonlonearity
    or a MLP with 1 hidden layer can learn XOR.
"""


def create_duplicated_xor_data(
    num_samples: int, 
    noise_std: float=0.1,
    num_nearest_neighbors: int=10,
    feature_repeats: int=5,):
    """
    This function creates XOR function features with duplicate features and optionally sparse features 
    using dropout, to make the XOR task more challenging

    Arguments:
    - num_samples:  Number of nodes to create, must be a number divisible by 4 (to balance XOR samples)
    - noise_std:    Standard deviation of noise that will be added to node features to make them "fuzzy"
    - same_class_link_prob:     Probability of linking nodes of the same class
    - diff_class_link_prob:     Probability of linking nodes of different classes
    - feature_repeats:  number of times to duplicate the two XOR features
    - dropout_rate:     dropout rate to use to make XOR duplicated features sparse

    Returns:
    - x: matrix of node features, shape [num_samples, 2]
    - labels: vector of corresponding labels, shape [num_samples,]
    """
    # Input validation
    assert num_samples % 4 == 0, "num_samples must be an integer divisible by 4."
    # assert same_class_link_prob < 1. and same_class_link_prob >= 0., "same_class_link_prob must be a float percent between 0 and 1."
    # assert diff_class_link_prob < 1. and diff_class_link_prob >= 0., "diff_class_link_prob must be a float percent between 0 and 1."
    repeats = num_samples // 4

    # Create node features
    x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y = np.array([0.,1.,1.,0.])
    x = np.repeat(x, [repeats, repeats, repeats, repeats], axis=0)
    x = np.tile(x, [1, feature_repeats])  # Duplicate features
    y = np.repeat(y, [repeats, repeats, repeats, repeats], axis=0)
    # x_t = F.dropout(x_t, p=dropout_rate, training=True)  # Dropout with given probability

    noise_matrix = np.random.normal(loc=0.0, scale=noise_std, size=(x.shape[0], x.shape[1]))
    x += noise_matrix

    # Uncomment to add noise with different std to each feature duplicate
    # noise_std_steps = np.arange(0.05, 0.55, 0.5 / feature_repeats)
    # for idx, noise_std_step in enumerate(noise_std_steps):
    #     noise_matrix = np.random.normal(loc=0.0, scale=noise_std_step, size=(x.shape[0], x.shape[1] // feature_repeats))
    #     noise_matrix_t = torch.tensor(noise_matrix, dtype=torch.float32)
    #     x_t[:, idx*2:idx*2+2] += noise_matrix_t  # Add small noise to differentiate repeats of 4 orignal samples

    nbrs = NearestNeighbors(n_neighbors=num_nearest_neighbors + 1, algorithm='ball_tree', metric='minkowski').fit(x)
    distances, indices = nbrs.kneighbors(x)

    # Create binary adjacency matrix
    adj_matrix = np.zeros(shape=(num_samples, num_samples), dtype=np.uint8)
    for row in range(indices.shape[0]):
        for col in range(0, indices.shape[1]):  # ToDo: Note, previous mistake was ignoring self-loops by starting at idx 1
            adj_matrix[row,indices[row,col]] = 1
    # for row in range(num_samples):
    #     for col in range(num_samples):
    #         if row == col:
    #             pass  # No Self Loops
    #         elif y[row] == y[col]:
    #             if random.random() < same_class_link_prob:
    #                 adj_matrix[row,col] = 1
    #         elif y[row] != y[col]:
    #             if random.random() < diff_class_link_prob:
    #                 adj_matrix[row,col] = 1
    #         else:
    #             raise Exception("Unknown XOR sample")

    # Create PyG edge index tensor shape [2, num_edges] from adjacency matrix
    source_idx_list = []
    dest_idx_list = []
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix[row])):
            if adj_matrix[row][col] > 0:
                source_idx_list.append(row)
                dest_idx_list.append(col)
    
    edge_idx_arr = np.array([source_idx_list, dest_idx_list])
    x_t = torch.tensor(x, dtype=torch.float32)
    return x_t, torch.tensor(y, dtype=torch.float32), torch.from_numpy(adj_matrix), torch.LongTensor(edge_idx_arr)


def create_xor_data(
    num_samples: int, 
    noise_std: float=0.1, 
    same_class_link_prob: float=0.7, 
    diff_class_link_prob: float=0.1):
    """
    This function creates a toy synthetic dataset where samples resemble a fuzzy XOR function.
    The goal is to create a reusable data-generation function which can create train/test graphs
    rapidly, for fast iteration on GCN and AMPNET.

    Arguments:
    - num_samples:  Number of nodes to create, must be a number divisible by 4 (to balance XOR samples)
    - noise_std:    Standard deviation of noise that will be added to node features to make them "fuzzy"
    - same_class_link_prob:     Probability of linking nodes of the same class
    - diff_class_link_prob:     Probability of linking nodes of different classes

    Returns:
    - x: matrix of node features, shape [num_samples, 2]
    - labels: vector of corresponding labels, shape [num_samples,]
    """

    # Input validation
    assert num_samples % 4 == 0, "num_samples must be an integer divisible by 4."
    assert same_class_link_prob < 1. and same_class_link_prob >= 0., "same_class_link_prob must be a float percent between 0 and 1."
    assert diff_class_link_prob < 1. and diff_class_link_prob >= 0., "diff_class_link_prob must be a float percent between 0 and 1."
    repeats = num_samples // 4

    # Create node features
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0.,1.,1.,0.])
    x = np.repeat(x, [repeats, repeats, repeats, repeats], axis=0)
    y = np.repeat(y, [repeats, repeats, repeats, repeats], axis=0)

    noise_matrix = np.random.normal(loc=0.0, scale=noise_std, size=(num_samples, 2))
    x = x + noise_matrix  # Add small noise to differentiate repeats of 4 orignal samples

    # Create binary adjacency matrix
    adj_matrix = np.zeros(shape=(num_samples, num_samples), dtype=np.uint8)
    for row in range(num_samples):
        for col in range(num_samples):
            if row == col:
                pass  # No Self Loops
            elif y[row] == y[col]:
                if random.random() < same_class_link_prob:
                    adj_matrix[row,col] = 1
            elif y[row] != y[col]:
                if random.random() < diff_class_link_prob:
                    adj_matrix[row,col] = 1
            else:
                raise Exception("Unknown XOR sample")

    # Create PyG edge index tensor shape [2, num_edges] from adjacency matrix
    source_idx_list = []
    dest_idx_list = []
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix[row])):
            if adj_matrix[row][col] > 0:
                source_idx_list.append(row)
                dest_idx_list.append(col)
    
    edge_idx_arr = np.array([source_idx_list, dest_idx_list])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.from_numpy(adj_matrix), torch.LongTensor(edge_idx_arr)


def plot_node_features(node_features, labels, save_path=None, save_name=None):
    plt.rcParams.update({'font.size': 18})
    scatter = plt.scatter(x=node_features[:,0], y=node_features[:,1], c=labels, cmap="winter")
    plt.title("Fuzzy XOR Node Features")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1], loc='center')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, save_name), facecolor="white", bbox_inches="tight")
    plt.close()


def plot_graph(adj_matrix, labels):
    G = nx.from_numpy_matrix(A=adj_matrix.numpy(), create_using=nx.DiGraph)  # DiGraph = directed graph
    green_edges = []  # Edges between nodes of same class
    red_edges = []  # Edges between nodes of different class
    for edge in G.edges():
        if labels[edge[0]] == labels[edge[1]]:
            green_edges.append(edge)
        else:
            red_edges.append(edge)
    
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('coolwarm'), 
                       node_color = labels, node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color='g', arrows=True)    
    print("Done")


if __name__ == "__main__":
    # For debugging purposes
    # x, y, adj_matrix, edge_idx_arr = create_xor_data(num_samples=20, noise_std=0.3, same_class_link_prob=0.7, diff_class_link_prob=0.1)
    x, y, adj_matrix, edge_idx_arr = create_duplicated_xor_data(
        num_samples=400, 
        noise_std=0.3,
        num_nearest_neighbors=10,
        feature_repeats=1
    )
    print("Node features:\n", x, "\n")
    print("Labels:\n", y, "\n")
    print("Adjacency Matrix:\n", adj_matrix, "\n")
    plot_node_features(node_features=x, labels=y, save_path="./", save_name="xor_node_feat_visual.png")
    plot_graph(adj_matrix, labels=y)
