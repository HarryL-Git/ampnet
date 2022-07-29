import os
import time
import multiprocessing as mp
from datetime import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from synthetic_benchmark.synthetic_training_modular import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_search_figure(expt_dropout_val_list, expt_acc_list, acc_type_list, fig_save_path):
    visual_df = pd.DataFrame({
        "Dropout Value": expt_dropout_val_list,
        "Accuracy": expt_acc_list,  
        "Train-test": acc_type_list
    })
    sns.boxplot(x="Dropout Value", y="Accuracy", hue="Train-test", data=visual_df, palette="time")
    plt.savefig(os.path.join(fig_save_path, "dropout_search_figure.png"), facecolor="white", bbox_inches="tight")
    plt.close()


process_results = []
def collect_results(result):
    global process_results
    process_results.append(result)


def run_experiment(save_path, args, process_idx):
    expt_save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    expt_grads_path = os.path.join(expt_save_path, "gradients")
    expt_activ_path = os.path.join(expt_save_path, "activations")

    if not os.path.exists(expt_save_path):
        os.mkdir(expt_save_path)
        os.system("cp ./synthetic_benchmark/synthetic_training.py {}/".format(expt_save_path))
        # os.system("touch {}".format(os.path.join(expt_save_path, "_details.txt")))
        with open(os.path.join(expt_save_path, "_details.txt"), "w") as logfile:
            logfile.write("Training with dropout rate {}".format(args["dropout"]))
    if not os.path.exists(expt_grads_path):
        os.mkdir(expt_grads_path)
    if not os.path.exists(expt_activ_path):
        os.mkdir(expt_activ_path)

    print("Training with dropout {}".format(args["dropout"]))
    start_time = time.time()
    max_train_accuracy, max_test_accuracy = train_model(
        args=args, 
        save_path=expt_save_path, 
        grads_path=expt_grads_path, 
        activ_path=expt_activ_path)
    print("Training took {} minutes.".format((time.time() - start_time) / 60.))
    print("Max train acc {}, max test acc {}".format(max_train_accuracy, max_test_accuracy), "\n")

    # Return tuple of results which can be sorted with other experiment process results and processed later
    return (process_idx, (args["dropout"], max_train_accuracy, "Train"), (args["dropout"], max_test_accuracy, "Test"))


def controller(save_path, args):
    """
    This function is a controller for searching over the hyperparameter space of a chosen
    hyperparameter.

    1. Dropout percentage
    2. Number of samples
    3. Probability of linking heterogenous nodes
    4. Learning rate
    """

    # Testing dropout values:
    # Define variables
    dropout_values = [0.7, 0.8, 0.9]  # , 0.25, 0.5

    # Multiprocessing
    num_cpus = mp.cpu_count() - 2
    counter = 0
    with mp.Pool(processes=num_cpus) as pool:
        for dropout_val in dropout_values:
            args["dropout"] = dropout_val
            for _ in range(args["experiment_repeats"]):
                pool.apply_async(
                    run_experiment, 
                    args=(save_path, args, counter), 
                    callback=collect_results)
                counter += 1
                time.sleep(5)  # Sleep for 5 seconds so that experiments have different datetimes
        pool.close()
        pool.join()
    # run_experiment(save_path, args, 0)  # For debugging

    # Sort gathered process results
    process_results.sort(key=lambda x: x[0])
    
    # Go through results and create lists for plotting
    expt_dropout_val_list = []
    expt_acc_list = []
    acc_type_list = []

    for result in process_results:
        # result[0] is process index, result[1] is training result, result[2] is test result
        expt_dropout_val_list.append(result[1][0])
        expt_acc_list.append(result[1][1])
        acc_type_list.append(result[1][2])
        expt_dropout_val_list.append(result[2][0])
        expt_acc_list.append(result[2][1])
        acc_type_list.append(result[2][2])
    
    plot_search_figure(expt_dropout_val_list, expt_acc_list, acc_type_list, fig_save_path=save_path)


def main():
    # Arguments
    ARGS = {
        "diff_class_link_prob": 0.05,
        "dropout": 0.3,
        "epochs": 200,
        "experiment_repeats": 20,
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
    
    # Create overall directory
    save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d-%H_%M_%S') + "_search")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Run experiments
    start_time = time.time()
    controller(save_path=save_path, args=ARGS)
    print("\nTotal search time was {:.3f} minutes".format((time.time() - start_time) / 60.))

if __name__ == "__main__":
    main()
