import numpy as np


incoming_edge_indices = np.where(graph_data.edge_index[1,:] == 0)[0]


incoming_node_indices = graph_data.edge_index[0,incoming_edge_indices]
incoming_node_labels = graph_data.y[graph_data.edge_index[0,incoming_edge_indices]]

incoming_edge_attentions = edge_attn_weights_matrix[incoming_edge_indices,:,:]

graph_data.x[incoming_node_indices[0]]



"""
edge_attn_weights_matrix.shape
graph_data

graph_data.edge_index.shape

incoming_edge_indices = np.where(graph_data.edge_index[1,:] == 0)[0]
incoming_node_indices = graph_data.edge_index[0,incoming_edge_indices]
incoming_node_labels = graph_data.y[graph_data.edge_index[0,incoming_edge_indices]]
incoming_node_indices.shape

incoming_edge_attentions = edge_attn_weights_matrix[incoming_edge_indices,:,:]
incoming_edge_attentions.shape


incoming_edge_attentions[0,:,:]
graph_data.x[incoming_node_indices[0]]
"""


"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

classes = [0, 1]
scatter = ax.scatter3D(x[:,0].detach().numpy(), x[:,1].detach().numpy(), x[:,2].detach().numpy(), c=data.y, cmap="winter")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.set_title("Separation of Average-Pooled Vectors")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
"""
