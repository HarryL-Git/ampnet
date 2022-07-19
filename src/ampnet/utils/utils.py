import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(v1, v2):
    return (v1 == v2).sum() / v1.shape[0]


def plot_loss_curves(train_losses, val_losses, epoch_count, save_path, model_name):
    assert len(train_losses) == len(val_losses) == epoch_count, "Unequal sizes in loss curve plotting."
    time = list(range(epoch_count))
    visual_df = pd.DataFrame({
        "Train Loss": train_losses,
        "Test Loss": val_losses,
        "Iteration": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iteration', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Iteration'], value_name="Loss Value", var_name="Loss Type"))
    plt.title("{} Loss Curves".format(model_name))
    plt.yscale("log")
    filename = "train_val_loss_curves"
    plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.close()


def plot_acc_curves(train_accs, val_accs, epoch_count, save_path, model_name):
    assert len(train_accs) == len(val_accs) == epoch_count, "Unequal sizes in accuracy curve plotting."
    time = list(range(epoch_count))
    visual_df = pd.DataFrame({
        "Train Accuracy": train_accs,
        "Test Accuracy": val_accs,
        "Iteration": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iteration', y='Accuracy Value', hue='Accuracy Type', data=pd.melt(visual_df, ['Iteration'], value_name="Accuracy Value", var_name="Accuracy Type"))
    plt.title("{} Accuracy Curves".format(model_name))
    filename = "train_val_accuracy_curves"
    plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.close()