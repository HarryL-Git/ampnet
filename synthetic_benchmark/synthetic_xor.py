import numpy as np

"""
XOR Rules
Feature1 - Feature2 - XOR
0 - 0 - 0
0 - 1 - 1
1 - 0 - 1
1 - 1 - 0

Notes:
- Not linearly separable. A linear layer itself cannot learn it, but a linear layer with a nonlonearity
    or a MLP with 1 hidden layer can learn XOR.
"""


def create_data(num_samples: int, same_class_link_prob: float=0.8, diff_class_link_prob: float=0.2):
    """
    This function creates a toy synthetic dataset where samples resemble a fuzzy XOR function.
    The goal is to create a reusable data-generation function which can create train/test graphs
    rapidly, for fast iteration on GCN and AMPNET.

    Arguments:
    - num_samples:  Number of nodes to create, must be a number divisible by 4 (to balance XOR samples)
    """

    # Input validation
    assert num_samples % 4 == 0, "num_samples must be an integer divisible by 4."
    pass



if __name__ == "__main__":
    # For debugging purposes
    create_data(num_samples=12)
