import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from synthetic_benchmark.xor_training_utils import get_xor_data, get_model


def plot_attn_weights(edge_attn_weights_matrix, graph_data, fig_save_path):
    """
    This is a utility function for plotting attention weights for the synthetic XOR dataset.
    It accepts a matrix of edge attention weight, and plots (for now) four distribution
    plots

    for idx in range(graph_data.edge_index.shape[1])

    Args:
    - edge_attn_weights_matrix: matrix of size [num_edges, 2, 2]
    """
    # Get lists of edge indices for homogenous and heterogenous edges
    homog_edge_indices = []
    heterog_edge_indices = []
    for idx in range(graph_data.edge_index.shape[1]):
        if graph_data.y[graph_data.edge_index[0,idx]] == graph_data.y[graph_data.edge_index[1,idx]]:
            homog_edge_indices.append(idx)
        else:
            heterog_edge_indices.append(idx)

    homog_visual_df = pd.DataFrame({
        "Feat 1 - Feat 1 Coefficients": edge_attn_weights_matrix[homog_edge_indices,0,0],
        "Feat 1 - Feat 2 Coefficients": edge_attn_weights_matrix[homog_edge_indices,0,1],
        "Feat 2 - Feat 1 Coefficients": edge_attn_weights_matrix[homog_edge_indices,1,0],
        "Feat 2 - Feat 2 Coefficients": edge_attn_weights_matrix[homog_edge_indices,1,1]
    })
    homog_visual_df_melted = pd.melt(homog_visual_df, var_name="Relationship")

    heterog_visual_df = pd.DataFrame({
        "Feat 1 - Feat 1 Coefficients": edge_attn_weights_matrix[heterog_edge_indices,0,0],
        "Feat 1 - Feat 2 Coefficients": edge_attn_weights_matrix[heterog_edge_indices,0,1],
        "Feat 2 - Feat 1 Coefficients": edge_attn_weights_matrix[heterog_edge_indices,1,0],
        "Feat 2 - Feat 2 Coefficients": edge_attn_weights_matrix[heterog_edge_indices,1,1]
    })
    heterog_visual_df_melted = pd.melt(heterog_visual_df, var_name="Relationship")

    sns.set_theme()
    g = sns.FacetGrid(homog_visual_df_melted, col="Relationship", col_wrap=2, sharex=True, height=4)
    g.map(plt.hist, "value", alpha=.4)
    g.set_ylabels('Count')
    plt.savefig(os.path.join(fig_save_path, "homogenous_edge_attn_coeff_grid.png"), bbox_inches="tight", facecolor="white")
    plt.close()

    sns.set_theme()
    g = sns.FacetGrid(heterog_visual_df_melted, col="Relationship", col_wrap=2, sharex=True, height=4)
    g.map(plt.hist, "value", alpha=.4)
    g.set_ylabels('Count')
    plt.savefig(os.path.join(fig_save_path, "heterogenous_edge_attn_coeff_grid.png"), bbox_inches="tight", facecolor="white")
    plt.close()


def visualize_attention_coefficients(args, save_path):
    # Define model - load from ran experiment
    model = get_model(args["model_name"], args["dropout"])
    checkpoint = torch.load(os.path.join(args["experiment_load_dir_path"], args["experiment_load_name"], "final_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define data
    train_data, test_data = get_xor_data(
        num_samples=args["num_samples"], 
        noise_std=args["noise_std"], 
        same_class_link_prob=args["same_class_link_prob"], 
        diff_class_link_prob=args["diff_class_link_prob"], 
        save_path=save_path)

    _ = model(train_data)
    print(model.conv1.attn_output_weights.shape)
    print(model.conv2.attn_output_weights.shape)

    plot_attn_weights(
        edge_attn_weights_matrix=model.conv1.attn_output_weights.detach().numpy(), 
        graph_data=train_data,
        fig_save_path=save_path
    )


def main():
    # Arguments
    args = {
        "diff_class_link_prob": 0.05,
        "dropout": 0.0,
        "epochs": 200,
        "experiment_load_dir_path": "./synthetic_benchmark/runs_AMPNet/2022-08-02-12_17_34_search",
        "experiment_load_name": "2022-08-02-12_20_45_7_0.8",
        "gradient_activ_save_freq": 50,
        "learning_rate": 0.01,
        "model_name": "AMPNet",
        "noise_std": 0.3,
        "num_samples": 400,
        "same_class_link_prob": 0.1,
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

