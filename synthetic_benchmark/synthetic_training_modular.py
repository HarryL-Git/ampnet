import os
import time
import datetime

import torch
import torch.nn as nn
from src.ampnet.utils.utils import *
from synthetic_benchmark.xor_training_utils import get_xor_data, get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(args, save_path, grads_path, activ_path):
    # Define model
    model = get_model(args["model_name"], args["dropout"])
    model.to(device)

    # Define data
    train_data, test_data = get_xor_data(
        num_samples=args["num_samples"], 
        noise_std=args["noise_std"], 
        same_class_link_prob=args["same_class_link_prob"], 
        diff_class_link_prob=args["diff_class_link_prob"], 
        save_path=save_path)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    criterion = nn.MSELoss()
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(args["epochs"]):
        model.train()
        optimizer.zero_grad()

        out = model(train_data)
        train_loss = criterion(out.squeeze(-1), train_data.y)
        pred = (out.squeeze(-1) > 0.5).float()
        train_accuracy = accuracy(pred.detach().numpy(), train_data.y.detach().numpy())

        train_loss.backward()
        if epoch % 4 == 0:
            model.plot_grad_flow(grads_path, epoch, iter=0)
            model.visualize_gradients(grads_path, epoch, iter=0)
            model.visualize_activations(activ_path, train_data, epoch, iter=0)
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

    plot_loss_curves(
        train_loss_list, 
        test_loss_list, 
        epoch_count=args["epochs"], 
        save_path=save_path, 
        model_name=args["model_name"])
    plot_acc_curves(
        train_acc_list, 
        test_acc_list, 
        epoch_count=args["epochs"], 
        save_path=save_path, 
        model_name=args["model_name"])
    return max(train_acc_list), max(test_acc_list)


if __name__ == "__main__":
    # Arguments
    ARGS = {
        "diff_class_link_prob": 0.05,
        "dropout": 0.2,
        "epochs": 200,
        "learning_rate": 0.01,
        "model_name": "GCN",
        "noise_std": 0.05,
        "num_samples": 40,
        "same_class_link_prob": 0.8,
    }
    assert ARGS["model_name"] in ["LinearLayer", "TwoLayerSigmoid", "GCN", "AMPNet"]

    # Create save paths
    save_path = "./synthetic_benchmark/runs_{}".format(ARGS["model_name"])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
    ACTIV_PATH = os.path.join(SAVE_PATH, "activations")

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))
        os.system("cp ./synthetic_benchmark/synthetic_training.py {}/".format(SAVE_PATH))
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
