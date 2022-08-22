import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synthetic_benchmark.xor_training_utils import get_xor_data, get_duplicated_xor_data, get_model


def plot_attn_weights(edge_attn_weights_matrix, graph_data, fig_save_path):
    """
    This is a utility function for plotting attention weights for the synthetic XOR dataset.
    It accepts a matrix of edge attention weight, and plots (for now) four distribution
    plots

    # edge_attn_weights_matrix is shape (batch_size, target_sequence_len, source_seq_len)
        # --> row index is feature index of destination node
        # --> col index is feature index of source node
        # --> row sums up to one, because feature in destination node is contextualized 
        # by a linear combo with attention weights of source node features
    # Interpretation: attn_matr[idx, row, col] = how much does feature row_idx in destination
    # node listen to feature col_idx in source node

    Args:
    - edge_attn_weights_matrix: matrix of size [num_edges, 2, 2]
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
        if graph_data.edge_index[0,edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[0,edge_idx] >= cutoffs[1] and graph_data.edge_index[0,edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[0,edge_idx] >= cutoffs[2] and graph_data.edge_index[0,edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[0,edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown origin XOR node")
        
        edge_connection_str += "-"

        if graph_data.edge_index[1,edge_idx] < cutoffs[1]:
            edge_connection_str += "00"
        elif graph_data.edge_index[1,edge_idx] >= cutoffs[1] and graph_data.edge_index[1,edge_idx] < cutoffs[2]:
            edge_connection_str += "01"
        elif graph_data.edge_index[1,edge_idx] >= cutoffs[2] and graph_data.edge_index[1,edge_idx] < cutoffs[3]:
            edge_connection_str += "10"
        elif graph_data.edge_index[1,edge_idx] >= cutoffs[3]:
            edge_connection_str += "11"
        else:
            raise Exception("Unknown destination XOR node")
        
        df_dict[edge_connection_str].append(edge_idx)
    
    for key in list(df_dict.keys()):
        if len(df_dict[key]) == 0:
            continue
        edge_coeffs_df = pd.DataFrame({
            "Dst Feat 1 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key],0,0],
            "Dst Feat 1 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key],0,1],
            "Dst Feat 2 attend to Src Feat 1": edge_attn_weights_matrix[df_dict[key],1,0],
            "Dst Feat 2 attend to Src Feat 2": edge_attn_weights_matrix[df_dict[key],1,1]
        })
        edge_coeffs_df_melted = pd.melt(edge_coeffs_df, var_name="Relationship")

        sns.set_theme()
        g = sns.FacetGrid(edge_coeffs_df_melted, col="Relationship", col_wrap=2, sharex=True, sharey=True, height=4)
        g.map(plt.hist, "value", alpha=.4, bins=np.arange(0.0, 1.05, 0.05))
        g.set_ylabels('Count')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(key)
        plt.savefig(os.path.join(fig_save_path, "{}_attn_coeff_grid.png".format(key)), bbox_inches="tight", facecolor="white")
        plt.close()


def visualize_attention_coefficients(args, save_path):
    # Define model - load from ran experiment
    model = get_model(args["model_name"], args["dropout"])
    checkpoint = torch.load(os.path.join(args["experiment_load_dir_path"], args["experiment_load_name"], "final_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define data
    if not args["use_duplicated_xor_features"]:
        train_data, test_data = get_xor_data(
            num_samples=args["num_samples"], 
            noise_std=args["noise_std"], 
            same_class_link_prob=args["same_class_link_prob"], 
            diff_class_link_prob=args["diff_class_link_prob"], 
            save_path=save_path)
    else:
        train_data, test_data = get_duplicated_xor_data(
            num_samples=args["num_samples"], 
            noise_std=args["noise_std"], 
            num_nearest_neighbors=args["num_nearest_neighbors"],
            feature_repeats=args["feature_repeats"],
            save_path=save_path,
        )

    _ = model(train_data)
    print(model.conv1.attn_output_weights.shape)

    plot_attn_weights(
        edge_attn_weights_matrix=model.conv1.attn_output_weights.detach().numpy(), 
        graph_data=train_data,
        fig_save_path=save_path
    )


def main():
    # Arguments
    args = {
        # "diff_class_link_prob": 0.05,
        "dropout": 0.0,
        "epochs": 200,
        "feature_repeats": 1,
        "experiment_load_dir_path": "./synthetic_benchmark/runs_AMPNet/2022-08-16-17_29_31_search",
        "experiment_load_name": "2022-08-16-17_38_50_113_0.6",
        "gradient_activ_save_freq": 50,
        "learning_rate": 0.01,
        "model_name": "AMPNet",
        "noise_std": 0.4,
        "num_nearest_neighbors": 10,
        "num_samples": 400,
        # "same_class_link_prob": 0.5,
        "use_duplicated_xor_features": True,
    }
    assert args["model_name"] in ["LinearLayer", "TwoLayerSigmoid", "GCN", "GCNOneLayer", "AMPNet"]

    # Create save paths
    save_path = "./synthetic_benchmark/visualize_AMPNet_attn_coeffs"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # Name save directory same as experiment we are visualizing for
    SAVE_PATH = os.path.join(save_path, args["experiment_load_name"])
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    visualize_attention_coefficients(args, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()

