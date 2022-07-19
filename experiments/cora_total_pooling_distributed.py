import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import math
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from src.ampnet.conv.amp_conv import AMPConv, AMPConvOneBlock
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils.dropout import dropout_adj
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN

from torch.nn.parallel import DistributedDataParallel as DDP


# Global Variables
TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN


def train(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

    device = torch.device('cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    all_data = dataset[0].to(device)

    if rank == 0:
        save_path = "./experiments/runs_distrib" if TRAIN_AMPCONV else "./experiments/runs_GCN_baseline_distrib"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
        GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
        ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
            os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))
            os.system("cp ./experiments/cora_total_pooling.py {}/".format(SAVE_PATH))
        if not os.path.exists(GRADS_PATH):
            os.mkdir(GRADS_PATH)
        if not os.path.exists(ACTIV_PATH):
            os.mkdir(ACTIV_PATH)

    if TRAIN_AMPCONV:
        model = AMPGCN().to(device)
    else:
        model = GCN().to(device)
    # model.to(rank)  # For GPU training

    ddp_model = DDP(model)
    loader = GraphSAINTRandomWalkSampler(all_data, batch_size=10, walk_length=150,
                                     num_steps=10, sample_coverage=100)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001, weight_decay=5e-4)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    epochs = 30
    for epoch in range(epochs):

        total_loss = total_examples = 0
        for idx, data_obj in enumerate(loader):
            model.train()
            optimizer.zero_grad()
            train_loss = 0.0
            test_loss = 0.0
            data = data_obj.to(device)

            # edge_weight = data.edge_norm * data.edge_weight
            out = model(data)
            train_loss = F.nll_loss(out, data.y, reduction='none')
            train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
            train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).numpy(), 
                                        data.y[data.train_mask].detach().numpy())
            
            train_loss.backward()
            if idx % 4 == 0 and rank == 0:
                model.plot_grad_flow(GRADS_PATH, epoch, idx)
                model.visualize_gradients(GRADS_PATH, epoch, idx)
                model.visualize_activations(ACTIV_PATH, data, epoch, idx)
            optimizer.step()
            total_loss += train_loss.item() * data.num_nodes
            total_examples += data.num_nodes

            ########
            # Test #
            ########
            model.eval()
            with torch.no_grad():
                test_loss = F.nll_loss(out, data.y, reduction='none')
                test_loss = (test_loss * data.node_norm)[data.test_mask].sum()
                test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).numpy(),
                                            data.y[data.test_mask].detach().numpy())

            print("Epoch {:05d} Partition {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} ".format(epoch, idx, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_accuracy)
            test_loss_list.append(test_loss.item())
            test_acc_list.append(test_accuracy)

    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    start_time = time.time()

    for rank in range(size):
        p = mp.Process(target=train, args=(rank, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
