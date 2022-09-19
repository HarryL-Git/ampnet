import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.datasets import Planetoid
from src.ampnet.module.amp_gcn import AMPGCN


os.chdir("..")  # Change current working directory to parent directory of GitHub repository


def plot_attn_weights(edge_attn_weights_matrix, graph_data, fig_save_path):
    """
    This is a utility function for plotting attention weights for the Cora dataset.
    It accepts a matrix of edge attention weight

    # edge_attn_weights_matrix is shape (batch_size, target_sequence_len=20, source_seq_len=20)
        # --> row index is feature index of destination node
        # --> col index is feature index of source node
        # --> row sums up to one, because feature in destination node is contextualized
        # by a linear combo with attention weights of source node features
    # Interpretation: attn_matr[idx, row, col] = how much does feature row_idx in destination
    # node listen to feature col_idx in source node

    Args:
    - edge_attn_weights_matrix: matrix of size [batch_size, target_sequence_len=20, source_seq_len=20]
    """
    # Get lists of edge indices for homogenous and heterogenous edges
    cutoffs = list(range(0, graph_data.x.shape[0], graph_data.x.shape[0] // 4))
    df_dict = {
        "XOR_00-00": [],
        "XOR_00-01": [],
        "XOR_00-10": [],
        "XOR_00-11": [],
        "XOR_01-00": [],
        "XOR_01-01": [],
        "XOR_01-10": [],
        "XOR_01-11": [],
        "XOR_10-00": [],
        "XOR_10-01": [],
        "XOR_10-10": [],
        "XOR_10-11": [],
        "XOR_11-00": [],
        "XOR_11-01": [],
        "XOR_11-10": [],
        "XOR_11-11": [],
    }
    for edge_idx in range(graph_data.edge_index.shape[1]):
        edge_connection_str = "XOR_"
        if graph_data.edge_index[0, edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[1] and graph_data.edge_index[0, edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[2] and graph_data.edge_index[0, edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown origin XOR node")

        edge_connection_str += "-"

        if graph_data.edge_index[1, edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[1] and graph_data.edge_index[1, edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[2] and graph_data.edge_index[1, edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown destination XOR node")

        df_dict[edge_connection_str].append(edge_idx)

    for key in list(df_dict.keys()):
        if len(df_dict[key]) == 0:
            continue
        edge_coeffs_df = pd.DataFrame({
            "Dst Feat 1 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 0, 0],
            "Dst Feat 1 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 0, 1],
            "Dst Feat 2 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 1, 0],
            "Dst Feat 2 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 1, 1]
        })
        edge_coeffs_df_melted = pd.melt(edge_coeffs_df, var_name="Relationship")

        sns.set_theme()
        g = sns.FacetGrid(edge_coeffs_df_melted, col="Relationship", col_wrap=2, sharex=True, sharey=True, height=4)
        g.map(plt.hist, "value", alpha=.4, bins=np.arange(-4.0, 4.05, 0.4))
        g.set_ylabels('Count')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(key)
        plt.savefig(os.path.join(fig_save_path, "{}_attn_coeff_grid.png".format(key)), bbox_inches="tight",
                    facecolor="white")
        plt.close()


def plot_attn_weights_duplicate_features(edge_attn_weights_matrix, graph_data, fig_save_path):
    """
    This is a utility function for plotting attention weights for the synthetic XOR dataset.
    It accepts a matrix of edge attention weight, and plots distribution plots.

    # edge_attn_weights_matrix is shape (batch_size, target_sequence_len, source_seq_len)
        # --> row index is feature index of destination node
        # --> col index is feature index of source node
        # --> row sums up to one, because feature in destination node is contextualized
        # by a linear combo with attention weights of source node features
    # Interpretation: attn_matr[idx, row, col] = how much does feature row_idx in destination
    # node listen to feature col_idx in source node

    Args:
    - edge_attn_weights_matrix: matrix of size [num_edges, num_features, num_features]
    """
    # Get lists of edge indices for homogenous and heterogenous edges
    cutoffs = list(range(0, graph_data.x.shape[0], graph_data.x.shape[0] // 4))
    df_dict = {
        "XOR_Src00-Dest00": [],
        "XOR_Src00-Dest01": [],
        "XOR_Src00-Dest10": [],
        "XOR_Src00-Dest11": [],
        "XOR_Src01-Dest00": [],
        "XOR_Src01-Dest01": [],
        "XOR_Src01-Dest10": [],
        "XOR_Src01-Dest11": [],
        "XOR_Src10-Dest00": [],
        "XOR_Src10-Dest01": [],
        "XOR_Src10-Dest10": [],
        "XOR_Src10-Dest11": [],
        "XOR_Src11-Dest00": [],
        "XOR_Src11-Dest01": [],
        "XOR_Src11-Dest10": [],
        "XOR_Src11-Dest11": [],
    }
    for edge_idx in range(graph_data.edge_index.shape[1]):
        edge_connection_str = "XOR_Src"
        if graph_data.edge_index[0, edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[1] and graph_data.edge_index[0, edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[2] and graph_data.edge_index[0, edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[0, edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown origin XOR node")

        edge_connection_str += "-Dest"

        if graph_data.edge_index[1, edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[1] and graph_data.edge_index[1, edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[2] and graph_data.edge_index[1, edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[1, edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown destination XOR node")

        df_dict[edge_connection_str].append(edge_idx)

    for key in list(df_dict.keys()):
        if len(df_dict[key]) == 0:
            continue
        edge_coeffs_df = pd.DataFrame({
            "Dst Feat 1 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 0, 0],
            "Dst Feat 1 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 0, 1],
            "Dst Feat 1 attend to Src Feat 3": edge_attn_weights_matrix[df_dict[key], 0, 2],
            "Dst Feat 1 attend to Src Feat 4": edge_attn_weights_matrix[df_dict[key], 0, 3],
            "Dst Feat 2 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 1, 0],
            "Dst Feat 2 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 1, 1],
            "Dst Feat 2 attend to Src Feat 3": edge_attn_weights_matrix[df_dict[key], 1, 2],
            "Dst Feat 2 attend to Src Feat 4": edge_attn_weights_matrix[df_dict[key], 1, 3],
            "Dst Feat 3 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 2, 0],
            "Dst Feat 3 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 2, 1],
            "Dst Feat 3 attend to Src Feat 3": edge_attn_weights_matrix[df_dict[key], 2, 2],
            "Dst Feat 3 attend to Src Feat 4": edge_attn_weights_matrix[df_dict[key], 2, 3],
            "Dst Feat 4 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key], 3, 0],
            "Dst Feat 4 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key], 3, 1],
            "Dst Feat 4 attend to Src Feat 3": edge_attn_weights_matrix[df_dict[key], 3, 2],
            "Dst Feat 4 attend to Src Feat 4": edge_attn_weights_matrix[df_dict[key], 3, 3],
        })
        edge_coeffs_df_melted = pd.melt(edge_coeffs_df, var_name="Relationship")

        sns.set_theme()
        g = sns.FacetGrid(edge_coeffs_df_melted, col="Relationship", col_wrap=4, sharex=True, sharey=True, height=4)
        g.map(plt.hist, "value", alpha=.4, bins=np.arange(-7.5, 7.55, 0.5))  #
        g.set_ylabels('Count')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(key)
        plt.savefig(os.path.join(fig_save_path, "{}_attn_coeff_grid.png".format(key)), bbox_inches="tight",
                    facecolor="white")
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
        downsample_feature_vectors=True,
        average_pooling_flag=True,
        dropout_rate=0.0,
        dropout_adj_rate=0.0,
        feature_repeats=None).to(device)
    checkpoint = torch.load(
        os.path.join(args["experiment_load_dir_path"], args["experiment_load_name"], "model_checkpoint_ep10.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define data
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    all_data = dataset[0]

    _ = model(all_data)
    print(model.conv1.attn_output_weights.shape)

    if args["feature_repeats"] == 1:
        plot_attn_weights(
            edge_attn_weights_matrix=model.conv1.attn_output_weights.detach().numpy(),
            graph_data=all_data,
            fig_save_path=save_path
        )
    else:
        plot_attn_weights_duplicate_features(
            edge_attn_weights_matrix=model.conv1.attn_output_weights.detach().numpy(),
            graph_data=all_data,
            fig_save_path=save_path
        )


def main():
    # Arguments
    args = {
        "experiment_load_dir_path": "./experiments/runs/",
        "experiment_load_name": "2022-09-19-02_58_22",
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

