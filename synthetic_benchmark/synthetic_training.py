import os
import math
import time
import datetime

import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN
from src.ampnet.module.linear_layer import LinearLayer
from src.ampnet.module.two_layer_sigmoid_mlp import TwoLayerSigmoid
from synthetic_benchmark.synthetic_xor import create_xor_data, plot_node_features


# Global variables
TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create save paths
save_path = "./synthetic_benchmark/runs" if TRAIN_AMPCONV else "./synthetic_benchmark/runs_GCN"
# save_path = "./synthetic_benchmark/runs_linear_layer"
if not os.path.exists(save_path):
    os.mkdir(save_path)

SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))  # Empty details file
    os.system("cp ./synthetic_benchmark/synthetic_training.py {}/".format(SAVE_PATH))

if not os.path.exists(GRADS_PATH):
    os.mkdir(GRADS_PATH)
if not os.path.exists(ACTIV_PATH):
    os.mkdir(ACTIV_PATH)


if TRAIN_AMPCONV:
    model = AMPGCN(device="cpu", 
        embedding_dim=3, 
        num_heads=1,
        num_node_features=2, 
        num_sampled_vectors=2,
        output_dim=1, 
        softmax_out=False, 
        feat_emb_dim=2, 
        val_emb_dim=1,
        downsample_feature_vectors=False,
        average_pooling_flag=True).to(device)
else:
    model = GCN(
        num_node_features=2, 
        num_sampled_vectors=2,
        output_dim=1, 
        softmax_out=False,
        feat_emb_dim=2,
        val_emb_dim=1,
        downsample_feature_vectors=False).to(device)
# model = TwoLayerSigmoid()
# model = LinearLayer()

# Define data
x, y, adj_matrix, edge_idx_arr = create_xor_data(num_samples=40, noise_std=0.05, same_class_link_prob=0.8, diff_class_link_prob=0.05)
train_data = Data(x=x, edge_index=edge_idx_arr, y=y)
plot_node_features(x, y, SAVE_PATH, "xor_train_node_features.png")
x, y, adj_matrix, edge_idx_arr = create_xor_data(num_samples=40, noise_std=0.05, same_class_link_prob=0.8, diff_class_link_prob=0.05)
test_data = Data(x=x, edge_index=edge_idx_arr, y=y)
plot_node_features(x, y, SAVE_PATH, "xor_test_node_features.png")


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
start_time = time.time()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(train_data)
    train_loss = criterion(out.squeeze(-1), train_data.y)
    pred = (out.squeeze(-1) > 0.5).float()
    train_accuracy = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())

    train_loss.backward()
    if epoch % 4 == 0:
        model.plot_grad_flow(GRADS_PATH, epoch, iter=0)
        model.visualize_gradients(GRADS_PATH, epoch, iter=0)
        model.visualize_activations(ACTIV_PATH, train_data, epoch, iter=0)
    optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        pred = (out.squeeze(-1) > 0.5).float()
        test_loss = criterion(out.squeeze(-1), test_data.y)
        test_accuracy = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())
    
    print("Epoch {:05d} | Train Loss {:.4f}; Acc {:.4f} | Test Loss {:.4f} | Acc {:.4f} "
            .format(epoch, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
    train_loss_list.append(train_loss.item())
    train_acc_list.append(train_accuracy)
    test_loss_list.append(test_loss.item())
    test_acc_list.append(test_accuracy)


print("Training took {} minutes.".format((time.time() - start_time) / 60.))
# plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="2LayerSigmoid")
# plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="2LayerSigmoid")
if TRAIN_AMPCONV:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="AMPConv")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="AMPConv")
else:
    plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="GCN")
    plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="GCN")

model.eval()
with torch.no_grad():
    out = model(test_data)
    pred = (out.squeeze(-1) > 0.5).float()
    acc = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())
print(f'Final Test Accuracy: {acc:.4f}')
