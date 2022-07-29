from torch_geometric.data import Data
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN
from src.ampnet.module.linear_layer import LinearLayer
from src.ampnet.module.two_layer_sigmoid_mlp import TwoLayerSigmoid
from synthetic_benchmark.synthetic_xor import create_xor_data, plot_node_features


def get_xor_data(num_samples, noise_std, same_class_link_prob, diff_class_link_prob, save_path):
    x, y, adj_matrix, edge_idx_arr = create_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        same_class_link_prob=same_class_link_prob, 
        diff_class_link_prob=diff_class_link_prob)
    train_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x, y, save_path, "xor_train_node_features.png")

    x, y, adj_matrix, edge_idx_arr = create_xor_data(
        num_samples=num_samples, 
        noise_std=noise_std, 
        same_class_link_prob=same_class_link_prob, 
        diff_class_link_prob=diff_class_link_prob)
    test_data = Data(x=x, edge_index=edge_idx_arr, y=y)
    plot_node_features(x, y, save_path, "xor_test_node_features.png")

    return train_data, test_data


def get_model(model_name, dropout):
    if model_name == "AMPNet":
        model = AMPGCN(
            device="cpu", 
            embedding_dim=3, 
            num_heads=1,
            num_node_features=2, 
            num_sampled_vectors=2,
            output_dim=1, 
            softmax_out=False, 
            feat_emb_dim=2, 
            val_emb_dim=1,
            downsample_feature_vectors=False,
            average_pooling_flag=True)
    elif model_name == "GCN":
        model = GCN(
            num_node_features=2, 
            num_sampled_vectors=2,
            output_dim=1, 
            softmax_out=False,
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
