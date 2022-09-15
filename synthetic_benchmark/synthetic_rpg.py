import numpy as np
import pandas as pd
import pickle
import argparse
import networkx as nx
import datetime
import random
import multiprocessing as mp
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic RGB Random Partition Graph Datasets')
    parser.add_argument('-D', '--dataset', type=str, default='Colors')
    parser.add_argument('-o', '--out_dir', type=str, default='./synthetic_benchmark/synthetic_RGB/data', help='set saving path for the dataset')
    parser.add_argument('--RGB_train', type=int, default=100, help='number of graphs for training')
    parser.add_argument('--RGB_valid', type=int, default=300, help='number of graphs as the the validation set')
    parser.add_argument('--RGB_test', type=int, default=300, help='number of graphs for testing')
    parser.add_argument('--Nodes_min', type=int, default=3, help='smallest number of nodes per class')
    parser.add_argument('--Nodes_max', type=int, default=10, help='largest number of nodes per class')
    parser.add_argument('--Homophily_min', type=float, default=0.5, help='minimum probability of connecting vertices within a group')
    parser.add_argument('--Homophily_max', type=float, default=0.9, help='maximum probability of connecting vertices within a group')
    parser.add_argument('--Heterophily_min', type=float, default=0.1, help='minimum probability of connecting vertices within a different group')
    parser.add_argument('--Heterophily_max', type=float, default=0.5, help='maximum probability of connecting vertices within a different group')
    parser.add_argument('--RGB_train_max', type=int, default=30, help='maximum number of total nodes per graph in the training set')
    parser.add_argument('--dim', type=int, default=3, help='dimensionality of node feature')
    parser.add_argument('--Max_index', type=int, default=255, help='max index value in a node feature vector')
    parser.add_argument('--seed', type=int, default=111, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args





def Random_Partition_Graph(N_groups, N_vertices, Homophily, Heterophily, seed=None, directed=False):
    """Return the random partition graph with a partition of sizes.

    A partition graph is a graph of communities with sizes defined by
    s in sizes. Nodes in the same group are connected with probability
    p_in and nodes of different groups are connected with probability
    p_out.

    Arguments:
    - N_groups: Number of groups in the graph (type = int)
    - N_vertices: Number of vertices in each group (type = int)
    - Homophily: Probability linking nodes of the same group (type = float)
    - Heterophily: Probability linking nodes of the different group (type = float)
    - seed: Indicator for the random number generator (type = int)
    - directed : Determine if the graph is directed or not (type = boolean)

    Returns:
    - G: N_groups-partition graph (NetworkX Graph or DiGraph)
    """

    sizes = N_groups * [N_vertices]
    assert seed is None, random.seed(seed)
    assert 0.0 <= Homophily <= 1.0, nx.NetworkXError("p_in must be in [0,1]")
    assert 0.0 <= Heterophily <= 1.0, nx.NetworkXError("p_out must be in [0,1]")

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.graph['partition'] = []
    n = sum(sizes)
    G.add_nodes_from(range(n))
    # start with len(sizes) groups of gnp random graphs with parameter p_in
    # graphs are unioned together with node labels starting at
    # 0, sizes[0], sizes[0]+sizes[1], ...
    next_group = {}  # maps node key (int) to first node in next group
    start = 0
    group = 0
    for n in sizes:
        edges = ((u+start, v+start)
                 for u, v in
                 nx.fast_gnp_random_graph(n, Homophily, directed=directed).edges())
        G.add_edges_from(edges)
        next_group.update(dict.fromkeys(range(start, start+n), start+n))
        G.graph['partition'].append(set(range(start, start+n)))
        group += 1
        start += n
    # handle edge cases
    if Heterophily == 0:
        return G
    if Heterophily == 1:
        for n in next_group:
            targets = range(next_group[n], len(G))
            G.add_edges_from(zip([n]*len(targets), targets))
            if directed:
                G.add_edges_from(zip(targets, [n]*len(targets)))
        return G
    # connect each node in group randomly with the nodes not in group
    # use geometric method like fast_gnp_random_graph()
    lp = math.log(1.0 - Heterophily)
    n = len(G)
    if directed:
        for u in range(n):
            v = 0
            while v < n:
                lr = math.log(1.0 - random.random())
                v += int(lr/lp)
                # skip over nodes in the same group as v, including self loops
                if next_group.get(v, n) == next_group[u]:
                    v = next_group[u]
                if v < n:
                    G.add_edge(u, v)
                    v += 1
    else:
        for u in range(n-1):
            v = next_group[u]  # start with next node not in this group
            while v < n:
                lr = math.log(1.0 - random.random())
                v += int(lr/lp)
                if v < n:
                    G.add_edge(u, v)
                    v += 1
    return G





def Set_RPG_Node_Features_Colors(N_groups, N_vertices, Homophily, Heterophily, Max_index = 255):
    G = Random_Partition_Graph(N_groups = N_groups, N_vertices = N_vertices, Homophily = Homophily, Heterophily = Heterophily)
    G_array = nx.to_numpy_array(G, dtype=int)
    feature_list = np.zeros((3,1))        # set initinal feature value of three colors to zero
    Total_nodes = N_groups * N_vertices
    i = 0
    while i < Total_nodes:
        Edges = sum(G_array[i])     # find number of edges for specific node
        Indexs = Max_index/ Edges   # equaliy weighted Man_index to each existing edges for selected node
        source_list = G_array[i]    # Create list of edges for selected node
        Red = []                    # Create list for red color index
        Green = []                  # Creat a list for green color index
        Blue = []
        for index in range(0, N_vertices):
            Red.append(source_list[index])
            feature_list[0] = Indexs * sum(Red)   # get the index for red
        for index in range(N_vertices, 2 * N_vertices):
            Green.append(source_list[index])
            feature_list[1] = Indexs * sum(Green) # get the index for green
        for index in range(2 * N_vertices, 3 * N_vertices):
            Blue.append(source_list[index])
            feature_list[2] = Indexs * sum(Blue) # get the index for blue
        G.nodes[i]['color'] = feature_list       # assign all three color index to a seleced node
        feature_list = np.zeros((3,1))           # reset the feature list to zero
        i = i + 1
    return G