import random

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from src.ampnet.utils.utils import *
from src.ampnet.module.amp_gcn import AMPGCN

os.chdir("..")  # Change current working directory to parent directory of GitHub repository
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0]


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

# batch size 1, walk length 500 => ~225 nodes
# batch size 20, walk length 100 => ~750 nodes
# batch size 10, walk length 100 => ~500 nodes
num_steps = 200  # ToDo: Need to get 1000 iterations/batches
loader = GraphSAINTRandomWalkSampler(all_data, batch_size=8, walk_length=150,
                                     num_steps=num_steps, sample_coverage=100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2)

epochs = 1000 // num_steps  # 1000 iterations/batches total
for epoch in range(epochs):
    for idx, data_obj in enumerate(loader):
        print("Epoch {:05d}, iteration {:05d}, lr {:05f}, optim lr {:.05f}".format(epoch, idx, float(scheduler.get_last_lr()[0]), optimizer.param_groups[0]['lr']))
        model.train()
        optimizer.zero_grad()
        data = data_obj.to(device)

        out = model(data)

        train_loss = F.nll_loss(out, data.y.to(device), reduction='none')
        train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
        train_loss.backward()

        optimizer.step()
        scheduler.step()
