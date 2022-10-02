import torch
import numpy as np
import torch_geometric.nn


class SimpleMessagePass(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')  # Mean aggregation of neighbor messages => averaging
        pass

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_i, x_j):
        """
        Pass messages from nodes x_j to nodes x_i.
        """
        return x_j  # Pass features of source node as message


def main(include_self_loop):
    x_data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10], [11, 11, 11]])  # shape [5, 3] - 5 nodes
    if include_self_loop:
        edge_arr = [
            [0, 1, 2, 3, 4],
            [2, 2, 2, 2, 2]
        ]  # All nodes connecting to 3rd node, no self loop
    else:
        edge_arr = [
            [0, 1, 3, 4],
            [2, 2, 2, 2]
        ]  # All nodes connecting to 3rd node, self loop included for node 3
    edge_arr = torch.tensor(edge_arr).long()
    x_data = torch.tensor(x_data).float()

    # Expectation:
    # Without self-loop, node 3's representation should solely be average of neighbors: [24 / 4 = 6, 6, 6]
    # With self-loops, node 3's representation should be average of all 5 nodes: [27 / 5 = 5.4, 5.4, 5.4]
    # All other nodes should be [0,0,0], no one will pass messages to them
    model = SimpleMessagePass()
    with torch.no_grad():
        out = model(x_data, edge_arr)

    print("Expt: include_self_loop is set to {}...".format(include_self_loop))
    print(out[2])


if __name__ == "__main__":
    main(include_self_loop=False)
    main(include_self_loop=True)
