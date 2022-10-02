import math
import random
import argparse

import cellpylib as cpl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from cellpylib.ctrbl_rule import CTRBLRule


def create_multicolor_cyclic_cellular_automata_graph(num_colors=6, grid_size=30, num_timesteps=32):
    # initialize a 60x60 2D cellular automaton
    # cellular_automaton = cpl.init_simple2d(60, 60)
    cellular_automaton = cpl.init_random2d(
        rows=grid_size,
        cols=grid_size,
        k=num_colors,
    )

    # Specify table rules with CTRBLRule. {5-tuple:next_state_val}, needs to be exhaustive
    # key = (current_activity, top, right, bottom, left)
    rule_table = {}
    for curr_val in range(num_colors):
        for top_val in range(num_colors):
            for right_val in range(num_colors):
                for bottom_val in range(num_colors):
                    for left_val in range(num_colors):
                        next_val = 0 if curr_val + 1 >= num_colors else curr_val + 1
                        val = next_val if next_val in [top_val, right_val, bottom_val, left_val] else curr_val
                        # If any neighbor has next val
                        rule_table[(curr_val, top_val, right_val, bottom_val, left_val)] = val
    ctrbl_rule = CTRBLRule(rule_table)

    #--- Evolve cellular automata for enough time steps that it reaches stable state ---#
    cellular_automaton = cpl.evolve2d(
        cellular_automaton,
        timesteps=1000,
        neighbourhood='von Neumann',  # 'von Neumann', 'Moore'
        apply_rule=ctrbl_rule)

    #--- Now evolve for the number of timesteps and concatenate grid at each timestep
    grid_states = []
    for _ in range(num_timesteps):
        cellular_automaton = cpl.evolve2d(
            cellular_automaton,
            timesteps=1,
            neighbourhood='von Neumann',  # 'von Neumann', 'Moore'
            apply_rule=ctrbl_rule)
        grid_states.append(cellular_automaton)

    grid_states = np.concatenate(grid_states, axis=0)  # Shape [num_timesteps, grid_size, grid_size]

    #--- Convert to a graph ---#
    # node_features shape [num_cells, num_timesteps]
    num_cells = grid_states.shape[1] * grid_states.shape[2]
    node_features = np.zeros(shape=(num_cells, grid_states.shape[0]), dtype=np.float32)
    adj_matrix = np.eye(num_cells, dtype=np.uint8)  # Start with Identity matrix for self connections
    for row_idx in range(grid_states.shape[1]):
        for col_idx in range(grid_states.shape[2]):
            cell_idx = row_idx * grid_states.shape[2] + col_idx
            # Set node features for cell
            node_features[cell_idx, :] = grid_states[:, row_idx, col_idx]

            # Set neighbor connections of cell
            left_neighbor = cell_idx - 1
            right_neighbor = cell_idx + 1
            top_neigbbor = cell_idx - grid_states.shape[2]
            bottom_neigbbor = cell_idx + grid_states.shape[2]

            if col_idx - 1 > 0:
                adj_matrix[row_idx, col_idx - 1] = 1
            if col_idx + 1 < grid_states.shape[2]:
                adj_matrix[row_idx, col_idx + 1] = 1
            if row_idx - 1 > 0:
                adj_matrix[row_idx - 1, col_idx] = 1
            if row_idx + 1 < grid_states.shape[1]:
                adj_matrix[row_idx + 1, col_idx] = 1

    # Create PyG edge idx array from adj matrix
    source_idx_list = []
    dest_idx_list = []
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix[row])):
            if adj_matrix[row][col] > 0:
                source_idx_list.append(row)
                dest_idx_list.append(col)

    edge_idx_arr = np.array([source_idx_list, dest_idx_list])

    return None  # ToDo: Fill here






def feature_embedding(node_features:np.ndarray):
  feature_list = np.zeros((len(node_features),3)) # Create a new array for RGB feature
  i = 0
  while i < len(node_features):
    # Using dictionary search color features from origional input
    unique, counts = np.unique(node_features[i], return_counts=True)
    node_color = dict(zip(unique, counts))

    Red = []                    # Create a list for red color index
    Green = []                  # Create a list for green color index
    Blue = []                   # Create a list for blue color index

    # get the total number of color features found in a node
    if 0 in node_color:
      color_0 = node_color[0]
    else:
      color_0 = 0
    if 1 in node_color:
      color_1 = node_color[1]
    else:
      color_1 = 0
    if 2 in node_color:
      color_2 = node_color[2]
    else:
      color_2 = 0
    if 3 in node_color:
      color_3 = node_color[3]
    else:
      color_3 = 0
    if 4 in node_color:
      color_4 = node_color[4]
    else:
      color_4 = 0   
    if 5 in node_color:
      color_5 = node_color[5]
    else:
      color_5 = 0

    # Calculate the R, G, B index based on the rule for each color (e.g color_0 = 150*R + 50*G + 55*B = 255, subject to change)
    Red.append(color_0*150 + color_1*180 + color_2*60 + color_3*75 + color_4*10 + color_5*40)
    Green.append(color_0*50 + color_1*300 + color_2*170 + color_3*160 + color_4*60 + color_5*80)
    Blue.append(color_0*55 + color_1*45  + color_2*25 + color_3*20 + color_4*185 + color_5*135)

    # assign RGB values to new feature list
    feature_list[i,0] = 255 * sum(Red)/(sum(Red)+ sum(Green) + sum(Blue))
    feature_list[i,1] = 255 * sum(Green)/(sum(Red)+ sum(Green) + sum(Blue))
    feature_list[i,2] = 255 * sum(Blue)/(sum(Red)+ sum(Green) + sum(Blue))

    i = i+1
  return feature_list





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

