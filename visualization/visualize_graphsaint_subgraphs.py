
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import to_networkx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
all_data = dataset[0].to(device)

loader = GraphSAINTRandomWalkSampler(all_data, batch_size=100, walk_length=3,
                                     num_steps=10, sample_coverage=100)

g = to_networkx(all_data, to_undirected=True)
nx.draw(g)


for epoch in range(5):
    total_loss = total_examples = 0
    for idx, data_obj in enumerate(loader):
        data = data_obj.to(device)
        g = to_networkx(data, to_undirected=True)
        nx.draw(g)


