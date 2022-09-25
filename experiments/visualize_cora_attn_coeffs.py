import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.datasets import Planetoid
from src.ampnet.module.amp_gcn import AMPGCN


os.chdir("..")  # Change current working directory to parent directory of GitHub repository


def get_edge_indices_between_nodes(graph_data, class_src, class_dst):
    """
    Helper function to compute and return indices of all edges between nodes of class_src to
    nodes of class_dst.

    Args:
        graph_data: Cora dataset object
        class_src: Source node class
        class_dst: Destination node class

    Returns:
        edge_indices: List of edge indices
    """
    edge_indices = []
    for edge_idx in range(graph_data.edge_index.shape[1]):
        if graph_data.y[graph_data.edge_index[0, edge_idx]] == class_src and \
                graph_data.y[graph_data.edge_index[1, edge_idx]] == class_dst:
            edge_indices.append(edge_idx)

    return np.array(edge_indices)


def get_top_30_feature_idxs_for_class(class_idx, graph_data):
    """
    Helper function to get indices of 30 most often sampled features for nodes of class class_idx.
    Get most common features for class. Two ways to do this:
     - See what features were most sampled by AMPNet
         ampnet_sampled_feat_idxs = model.sampled_node_feat_indices  # ampnet_sampled_feat_idxs: [num_nodes,20]
     - Use ground truth, find actual most common features* - doing this, makes it deterministic visualizing multiple times

    Args:
        class_idx: Index of class of nodes to analyze
        graph_data: Cora dataset object: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
        model: AMPNet model

    Returns:
        feature_idxs: list of 30 feature indices
    """
    # Get node indices of all nodes of class class_idx, then get node features for each of the nodes
    node_indices = np.where((graph_data.y == class_idx))[0]
    class_node_features = graph_data.x[node_indices]  # class_node_features: torch.Size([num_class_nodes, 1433])

    # Count feature presence by summing across all node features
    feature_presence_counts = class_node_features.sum(dim=0)  # torch.Size([1433])
    feature_presence_counts = feature_presence_counts.cpu().numpy()

    # Return top 30
    top30_feature_idxs = np.argpartition(feature_presence_counts, -30)[-30:]
    # top30_features = feature_presence_counts[top30_feature_idxs]

    return top30_feature_idxs


def calculate_attn_heatmap(total_edge_attn_weights_matrix, ampnet_sampled_features, graph_data, src_class_top_30_features, dest_class_top_30_features, src_class_idx, dest_class_idx):
    """

    Args:
        total_edge_attn_weights_matrix: matrix of size [num_edges, target_sequence_len=20, source_seq_len=20]
        ampnet_sampled_features: array of size [num_nodes, 20]
        graph_data: Cora dataset object
        src_class_top_30_features: array of indices of size [30,]
        dest_class_top_30_features: array of indices of size [30,]
        src_class_idx: index of source node class
        dest_class_idx: index of destination node class

    Returns:
        heatmap: matrix of size 30 x 30 representing attention coefficients between two node classes.

    """
    # Select all edges between two node classes
    edge_idxs = get_edge_indices_between_nodes(graph_data, src_class_idx, dest_class_idx)
    edges = graph_data.edge_index[:, edge_idxs]  # size [2, num_class_edges]
    edge_attn_weights = total_edge_attn_weights_matrix[edge_idxs]  # size [num_class_edges, targ_seq_len, src_seq_len]
    src_sampled_features = ampnet_sampled_features[edges[0, :]]  # size [num_class_edges, 20]
    dst_sampled_features = ampnet_sampled_features[edges[1, :]]  # size [num_class_edges, 20]

    # Form heatmap; each cell should be average attention coefficient between two features
    heatmap = np.zeros(shape=(30, 30))
    heatmap_coefficient_counts = np.zeros(shape=(30, 30))
    for edge_idx in range(edge_attn_weights.shape[0]):
        for dst_idx, dst_sampled_feat in enumerate(dst_sampled_features[edge_idx]):
            for src_idx, src_samp_feat in enumerate(src_sampled_features[edge_idx]):
                if src_samp_feat in src_class_top_30_features and dst_sampled_feat in dest_class_top_30_features:
                    # This attention coefficient belongs in this heatmap.
                    sampled_row_idx = np.where(src_class_top_30_features == src_samp_feat)[0]
                    sampled_col_idx = np.where(dest_class_top_30_features == dst_sampled_feat)[0]
                    heatmap[sampled_row_idx, sampled_col_idx] += edge_attn_weights[edge_idx, dst_idx, src_idx]
                    heatmap_coefficient_counts[sampled_row_idx, sampled_col_idx] += 1
    heatmap = np.divide(heatmap, heatmap_coefficient_counts,
                        out=np.zeros_like(heatmap),
                        where=heatmap_coefficient_counts != 0)
    return heatmap


def plot_attn_coefficients(args, edge_attn_weights_matrix, sampled_node_feat_indices, graph_data, fig_save_path):
    """
    This is a utility function for plotting attention weights for the Cora dataset.
    It accepts a matrix of edge attention weight

    # edge_attn_weights_matrix is shape (num_edges, target_sequence_len=20, source_seq_len=20)
        # --> row index is feature index of destination node
        # --> col index is feature index of source node
        # --> row sums up to one, because feature in destination node is contextualized
        # by a linear combo with attention weights of source node features
    # Interpretation: attn_matr[idx, row, col] = how much does feature row_idx in destination
    # node listen to feature col_idx in source node

    Args:
        - args: arguments dictionary
        - edge_attn_weights_matrix: matrix of size [batch_size, target_sequence_len=20, source_seq_len=20]
        - sampled_node_feat_indices: matrix of size [num_nodes, 20]
        - graph_data: Cora dataset object
        - fig_save_path: directory to save figure in
    """
    # Find top 30 features of both classes
    src_class_top_30_features = get_top_30_feature_idxs_for_class(args["class_src"], graph_data)
    dest_class_top_30_features = get_top_30_feature_idxs_for_class(args["class_dest"], graph_data)

    # Get 30 x 30 attention heatmap between two classes
    attn_heatmap = calculate_attn_heatmap(
        total_edge_attn_weights_matrix=edge_attn_weights_matrix,
        ampnet_sampled_features=sampled_node_feat_indices,
        graph_data=graph_data,
        src_class_top_30_features=src_class_top_30_features,
        dest_class_top_30_features=dest_class_top_30_features,
        src_class_idx=args["class_src"],
        dest_class_idx=args["class_dest"]
    )

    # Save heatmap
    np.save(os.path.join(fig_save_path, "heatmap_arr_raw"), arr=attn_heatmap)

    # Plot raw heatmap
    # src_class_top_30_feat_labels = ["Feature {}".format(num) for num in src_class_top_30_features]
    # dest_class_top_30_feat_labels = ["Feature {}".format(num) for num in dest_class_top_30_features]

    sns.set_theme()
    sns.set(rc={'figure.figsize': (9, 7)})
    sns.heatmap(attn_heatmap,
                vmin=0,
                vmax=attn_heatmap.max(),
                xticklabels=dest_class_top_30_features,
                yticklabels=src_class_top_30_features)
    plt.title("Class {} - Class {} Top 30 Features Heatmap".format(args["class_src"], args["class_dest"]), fontsize=18)
    plt.ylabel("Source Node Feature", fontsize=16)
    plt.xlabel("Destination Node Feature", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, "top30_class{}_class{}_unclustered_attn_heatmap.png".format(args["class_src"], args["class_dest"])),
                bbox_inches="tight", facecolor="white")
    plt.close()

    # Plot clustered heatmap
    g = sns.clustermap(attn_heatmap,
                       vmin=0,
                       vmax=attn_heatmap.max(),
                       xticklabels=dest_class_top_30_features,
                       yticklabels=src_class_top_30_features)
    ax = g.ax_heatmap
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    ax.set_title("Class {} - Class {} Top 30 Features Clustered Heatmap".format(args["class_src"], args["class_dest"]), fontsize=18)
    ax.set_xlabel("Destination Node Feature")
    ax.set_ylabel("Source Node Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, "top30_class{}_class{}_clustered_attn_heatmap.png".format(args["class_src"], args["class_dest"])),
                bbox_inches="tight", facecolor="white")
    plt.close()

def visualize_attention_coefficients(args, save_path):
    # Define model - load from ran experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AMPGCN(
        device=device,
        embedding_dim=128,
        num_heads=4,
        num_node_features=1433,
        num_sampled_vectors=20,
        output_dim=7,
        softmax_out=True,
        feat_emb_dim=127,
        val_emb_dim=1,
        downsample_feature_vectors=True,  # Might need all 1433 here
        average_pooling_flag=True,
        dropout_rate=0.0,
        dropout_adj_rate=0.0,
        feature_repeats=None).to(device)
    checkpoint = torch.load(
        os.path.join(args["experiment_load_dir_path"], args["experiment_load_name"], "model_checkpoint_ep50.pth"),
        map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define data
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    all_data = dataset[0]

    with torch.no_grad():
        _ = model(all_data)
        print(model.conv1.attn_output_weights.shape)

    plot_attn_coefficients(
        args=args,
        edge_attn_weights_matrix=model.conv1.attn_output_weights.detach().numpy(),
        sampled_node_feat_indices=model.sampled_node_feat_indices,
        graph_data=all_data,
        fig_save_path=save_path
    )


def main():
    # Arguments
    args = {
        "experiment_load_dir_path": "./experiments/runs/",
        "experiment_load_name": "2022-09-19-03_08_36*",
        "class_src": 2,  # Source node class
        "class_dest": 5,  # Destination node class
    }

    # Create save paths
    save_path = "experiments/visualize_AMPNet_attn_coeffs"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Name save directory same as experiment we are visualizing for
    SAVE_PATH = os.path.join(save_path, args["experiment_load_name"])
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    visualize_attention_coefficients(args, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()

