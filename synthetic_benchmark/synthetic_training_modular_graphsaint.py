import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from src.ampnet.utils.utils import *
from synthetic_benchmark.xor_training_utils import get_xor_data, get_duplicated_xor_data, get_model

os.chdir("..")  # Change current working directory to parent directory of GitHub repository
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(args, save_path, grads_path, activ_path, logfile=None):
    # Define model
    model = get_model(args["model_name"], args["dropout"], args)
    model.to(device)

    # Freeze all but last linear classifier layer
    # total_params = len(list(model.parameters()))
    # for counter, p in enumerate(model.parameters()):
    #     if counter < total_params - 2:  # Last two parameters are the weight and bias of the linear layer
    #         p.requires_grad = False

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
    
    train_loader = GraphSAINTRandomWalkSampler(train_data, batch_size=10, walk_length=150,
                                         num_steps=args["epoch_iterations"], sample_coverage=100)
    test_loader = GraphSAINTRandomWalkSampler(test_data, batch_size=10, walk_length=150,
                                         num_steps=1, sample_coverage=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=5e-4)
    # criterion = nn.NLLLoss()
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(args["epochs"]):
        for idx, data_obj in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            out = model(data_obj)
            # train_loss = criterion(out, train_data.y.long())
            train_loss = F.nll_loss(out, data_obj.y.long(), reduction='none')
            train_loss = (train_loss * data_obj.node_norm).sum()
            # pred = (out.squeeze(-1) > 0.5).float()
            pred = torch.argmax(out, dim=1)
            train_accuracy = accuracy(pred.detach().numpy(), data_obj.y.detach().numpy())

            train_loss.backward()
            if epoch % args["gradient_activ_save_freq"] == 0:
                model.plot_grad_flow(grads_path, epoch, iter=0)
                # model.visualize_gradients(grads_path, epoch, iter=0)
                # model.visualize_activations(activ_path, data_obj, epoch, iter=0)
            optimizer.step()

            # Test
            model.eval()
            with torch.no_grad():
                for test_idx, test_data_obj in enumerate(test_loader):
                    out = model(test_data_obj)
                    # pred = (out.squeeze(-1) > 0.5).float()
                    pred = torch.argmax(out, dim=1)
                    # test_loss = criterion(out, test_data.y.long(), reduction="none")
                    test_loss = F.nll_loss(out, test_data_obj.y.long(), reduction='none')
                    test_loss = (test_loss * test_data_obj.node_norm).sum()
                    test_accuracy = accuracy(pred.detach().numpy(), test_data_obj.y.detach().numpy())
            
            epoch_acc_str = "Epoch {:05d} Partition {:05d} | Train Loss {:.4f}; Acc {:.4f} | Test Loss {:.4f} | Acc {:.4f} " \
                .format(epoch, idx, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy)
            if logfile is None:
                print(epoch_acc_str)
            else:
                logfile.write(epoch_acc_str + "\n")
                logfile.flush()
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_accuracy)
            test_loss_list.append(test_loss.item())
            test_acc_list.append(test_accuracy)

    plot_loss_curves(
        train_loss_list, 
        test_loss_list, 
        epoch_count=args["epochs"] * args["epoch_iterations"], 
        save_path=save_path, 
        model_name=args["model_name"])
    plot_acc_curves(
        train_acc_list, 
        test_acc_list, 
        epoch_count=args["epochs"] * args["epoch_iterations"], 
        save_path=save_path, 
        model_name=args["model_name"])

    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict(),
        'validation_loss': test_loss.item()
    }, os.path.join(save_path, "final_model.pth"))

    return max(train_acc_list), max(test_acc_list)


if __name__ == "__main__":
    # Arguments
    ARGS = {
        # "diff_class_link_prob": 0.05,
        "dropout": 0.0,
        "epochs": 15,
        "epoch_iterations": 10,
        "feature_repeats": 716,
        "gradient_activ_save_freq": 50,
        "learning_rate": 0.01,
        "model_name": "AMPNet",
        "noise_std": 0.3,
        "num_nearest_neighbors": 20,
        "num_samples": 400,
        # "same_class_link_prob": 0.8,
        "use_duplicated_xor_features": True,
    }
    assert ARGS["model_name"] in ["LinearLayer", "TwoLayerSigmoid", "GCN", "GCNOneLayer", "AMPNet"]

    # Create save paths
    save_path = "synthetic_benchmark/runs_{}".format(ARGS["model_name"])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
    ACTIV_PATH = os.path.join(SAVE_PATH, "activations")

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))
        os.system("cp synthetic_benchmark/synthetic_training_modular_graphsaint.py {}/".format(SAVE_PATH))
        os.system("cp synthetic_benchmark/xor_training_utils.py {}/".format(SAVE_PATH))
    if not os.path.exists(GRADS_PATH):
        os.mkdir(GRADS_PATH)
    if not os.path.exists(ACTIV_PATH):
        os.mkdir(ACTIV_PATH)

    start_time = time.time()
    max_train_accuracy, max_test_accuracy = train_model(
        args=ARGS, 
        save_path=SAVE_PATH, 
        grads_path=GRADS_PATH, 
        activ_path=ACTIV_PATH)
    print("Training took {} minutes.".format((time.time() - start_time) / 60.))
    print("Max train acc {}, max test acc {}".format(max_train_accuracy, max_test_accuracy))
