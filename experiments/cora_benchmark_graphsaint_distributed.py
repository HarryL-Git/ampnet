import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import time
import datetime

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from src.ampnet.utils.utils import *
from src.ampnet.module.gcn_classifier import GCN
from src.ampnet.module.amp_gcn import AMPGCN

from torch.nn.parallel import DistributedDataParallel as DDP


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Global Variables
TRAIN_AMPCONV = True  # If False, trains a simple 2-layer GCN


def train(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9001'
    dist.init_process_group(backend, rank=rank, world_size=size)

    device = torch.device('cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    all_data = dataset[0]

    if rank == 0:
        save_path = "./runs_distrib" if TRAIN_AMPCONV else "./runs_GCN_baseline_distrib"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
        GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
        ACTIV_PATH = os.path.join(SAVE_PATH, "activations")
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
            os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))
            os.system("cp ./cora_benchmark_graphsaint_distributed.py {}/".format(SAVE_PATH))
            if TRAIN_AMPCONV:
                os.system("cp ./src/ampnet/conv/amp_conv.py {}/".format(SAVE_PATH))
                os.system("cp ./src/ampnet/module/amp_gcn.py {}/".format(SAVE_PATH))
            else:
                os.system("cp ./src/ampnet/module/gcn_classifier.py {}/".format(SAVE_PATH))
        if not os.path.exists(GRADS_PATH):
            os.mkdir(GRADS_PATH)
        if not os.path.exists(ACTIV_PATH):
            os.mkdir(ACTIV_PATH)

    if TRAIN_AMPCONV:
        model = AMPGCN().to(device)
    else:
        model = GCN(device=device).to(device)
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
            data = data_obj.to(device)

            # edge_weight = data.edge_norm * data.edge_weight
            out = model(data)
            train_loss = F.nll_loss(out, data.y, reduction='none')
            train_loss = (train_loss * data.node_norm)[data.train_mask].sum()
            train_accuracy = accuracy(out[data.train_mask].argmax(dim=1).cpu().numpy(),
                                      data.y[data.train_mask].detach().cpu().numpy())
            
            train_loss.backward()
            if rank == 0 and idx % 4 == 0:
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
                test_accuracy = accuracy(out[data.test_mask].argmax(dim=1).cpu().numpy(),
                                         data.y[data.test_mask].detach().cpu().numpy())
            if rank == 0:
                print("Epoch {:05d} Partition {:05d} | Train NLL Loss {:.4f}; Acc {:.4f} | Test NLL Loss {:.4f}; Acc {:.4f} ".format(epoch, idx, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))
                train_loss_list.append(train_loss.item())
                train_acc_list.append(train_accuracy)
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_accuracy)
    
    if rank == 0:
        print("Training took {} minutes.".format((time.time() - start_time) / 60.))
        if TRAIN_AMPCONV:
            plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="AMPConv")
            plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="AMPConv")
        else:
            plot_loss_curves(train_loss_list, test_loss_list, epoch_count=len(train_loss_list), save_path=SAVE_PATH, model_name="GCN")
            plot_acc_curves(train_acc_list, test_acc_list, epoch_count=len(train_acc_list), save_path=SAVE_PATH, model_name="GCN")

    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    start_time = time.time()

    for rank in range(size):
        p = mp.Process(target=train, args=(rank, size, 'gloo'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    print("Training took {} minutes.".format((end_time - start_time) / 60.))
