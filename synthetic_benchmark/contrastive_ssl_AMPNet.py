import os
import time
import datetime

import torch
import torch.nn as nn
from src.ampnet.utils.utils import *
from synthetic_benchmark.xor_training_utils import get_xor_data, get_duplicated_xor_data, get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Collection of functions for skipgram-like loss for unsupervised training of GraphSAGE taken from
GraphSAGE tensorflow repository.

def affinity(inputs1, inputs2):
    # Affinity score between batch of inputs1 and inputs2.
    # Args:
    #     inputs1: tensor of shape [batch_size x feature_size].
    
    # shape: [batch_size, input_dim1]
    if self.bilinear_weights:
        prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
        self.prod = prod
        result = tf.reduce_sum(inputs1 * prod, axis=1)
    else:
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
    return result

def neg_cost(inputs1, neg_samples, hard_neg_samples=None):
        # For each input in batch, compute the sum of its affinity to negative samples.
        # Returns:
        #     Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
        #         negative samples is computed.
        
        if self.bilinear_weights:
            inputs1 = tf.matmul(inputs1, self.vars['weights'])
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        return neg_aff

def _skipgram_loss(inputs1, inputs2, neg_samples, hard_neg_samples=None):
    aff = self.affinity(inputs1, inputs2)  # First term in unsupervised loss
    neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)  # second term
    neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))  # reduce sum over negative samples?
    loss = tf.reduce_sum(aff - neg_cost)
    return loss
"""





def train_model(args, save_path, grads_path, activ_path, logfile=None):
    # Define model
    model = get_model(args["model_name"], args["dropout"])
    model.to(device)

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
            same_class_link_prob=args["same_class_link_prob"], 
            diff_class_link_prob=args["diff_class_link_prob"], 
            feature_repeats=5,
            dropout_rate=0.0,
            save_path=save_path,
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=5e-4)
    criterion = None  # Implement GraphSAGE unsupervised loss function

    for epoch in range(args["epochs"]):
        model.train()
        optimizer.zero_grad()

        out = model(train_data)
        train_loss = criterion(out, train_data.y.long())

        train_loss.backward()
        if epoch % args["gradient_activ_save_freq"] == 0:
            model.plot_grad_flow(grads_path, epoch, iter=0)
            # model.visualize_gradients(grads_path, epoch, iter=0)
            # model.visualize_activations(activ_path, train_data, epoch, iter=0)
        optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            out = model(test_data)
            test_loss = criterion(out, test_data.y.long())
        
        epoch_acc_str = "Epoch {:05d} | Train Loss {:.4f}| Test Loss {:.4f}" \
            .format(epoch, train_loss.item(), test_loss.item())
        if logfile is None:
            print(epoch_acc_str)
        else:
            logfile.write(epoch_acc_str + "\n")
            logfile.flush()



    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict(),
        'validation_loss': test_loss.item()
    }, os.path.join(save_path, "final_model.pth"))


if __name__ == "__main__":
    # Arguments
    ARGS = {
        "diff_class_link_prob": 0.05,
        "dropout": 0.0,
        "epochs": 200,
        "gradient_activ_save_freq": 50,
        "learning_rate": 0.01,
        "model_name": "AMPNet",
        "noise_std": 0.3,
        "num_samples": 400,
        "same_class_link_prob": 0.8,
        "use_duplicated_xor_features": True,
    }
    assert ARGS["model_name"] in ["LinearLayer", "TwoLayerSigmoid", "GCN", "GCNOneLayer", "AMPNet"]

    # Create save paths
    save_path = "./synthetic_benchmark/runs_{}".format(ARGS["model_name"] + "_contrastive_SSL")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    SAVE_PATH = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    GRADS_PATH = os.path.join(SAVE_PATH, "gradients")
    ACTIV_PATH = os.path.join(SAVE_PATH, "activations")

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.system("touch {}".format(os.path.join(SAVE_PATH, "_details.txt")))
        os.system("cp ./synthetic_benchmark/synthetic_training_modular.py {}/".format(SAVE_PATH))
    if not os.path.exists(GRADS_PATH):
        os.mkdir(GRADS_PATH)
    if not os.path.exists(ACTIV_PATH):
        os.mkdir(ACTIV_PATH)

    start_time = time.time()
    train_model(
        args=ARGS, 
        save_path=SAVE_PATH, 
        grads_path=GRADS_PATH, 
        activ_path=ACTIV_PATH)
    print("Training took {} minutes.".format((time.time() - start_time) / 60.))
