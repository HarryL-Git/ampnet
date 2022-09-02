from torch_geometric.data import Data
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.gcn_one_layer import GCNOneLayer
from src.ampnet.module.amp_gcn import AMPGCN
from src.ampnet.module.linear_layer import LinearLayer
from src.ampnet.module.two_layer_sigmoid_mlp import TwoLayerSigmoid
from synthetic_benchmark.synthetic_xor import create_xor_data, create_duplicated_xor_data, plot_node_features


def get_xor_data(num_samples, noise_std, same_class_link_prob, diff_class_link_prob, save_path):
    x, y, adj_matrix, edge_idx_arr = create_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        same_class_link_prob=same_class_link_prob, 
        diff_class_link_prob=diff_class_link_prob
    )
    train_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x, y, save_path, "xor_train_node_features.png")

    # Fixed number of test samples
    x, y, adj_matrix, edge_idx_arr = create_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        same_class_link_prob=same_class_link_prob, 
        diff_class_link_prob=diff_class_link_prob
    )
    test_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x, y, save_path, "xor_test_node_features.png")

    return train_data, test_data


def get_duplicated_xor_data(num_samples, noise_std, num_nearest_neighbors, feature_repeats, save_path):
    x, y, adj_matrix, edge_idx_arr = create_duplicated_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        num_nearest_neighbors=num_nearest_neighbors,
        feature_repeats=feature_repeats
    )
    train_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x[:,0:2], y, save_path, "xor_train_first_two_features.png")

    # Fixed number of test samples
    x, y, adj_matrix, edge_idx_arr = create_duplicated_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        num_nearest_neighbors=num_nearest_neighbors,
        feature_repeats=feature_repeats
    )
    test_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x[:,0:2], y, save_path, "xor_test_first_two_features.png")

    return train_data, test_data


def get_model(model_name, dropout, args):
    if model_name == "AMPNet":
        model = AMPGCN(
            device="cpu", 
            embedding_dim=32,  # 128
            num_heads=2,
            num_node_features=args["feature_repeats"] * 2,  # Needs to match with feature_repeats, if using XOR data. 1432
            num_sampled_vectors=20,  # 100
            output_dim=2, 
            softmax_out=True, 
            feat_emb_dim=31,  # 127 
            val_emb_dim=1,
            downsample_feature_vectors=True,  # True
            average_pooling_flag=True,
            dropout_rate=dropout,
            dropout_adj_rate=dropout,
            feature_repeats=args["feature_repeats"])  # 716
    elif model_name == "GCN":
        model = GCN(
            num_node_features=2, 
            hidden_dim=2,
            num_sampled_vectors=2,
            output_dim=2, 
            softmax_out=True,
            feat_emb_dim=2,
            val_emb_dim=1,
            downsample_feature_vectors=False,
            dropout_rate=dropout,
            dropout_adj_rate=dropout)
    elif model_name == "GCNOneLayer":
        model = GCNOneLayer(
            num_node_features=2, 
            num_sampled_vectors=2,
            output_dim=2, 
            softmax_out=True,
            feat_emb_dim=2,
            val_emb_dim=1,
            downsample_feature_vectors=False,
            dropout_rate=dropout,
            dropout_adj_rate=dropout)
    elif model_name == "LinearLayer":
        model = LinearLayer()
    elif model_name == "TwoLayerSigmoid":
        model = TwoLayerSigmoid()
    else:
        return None

    return model
