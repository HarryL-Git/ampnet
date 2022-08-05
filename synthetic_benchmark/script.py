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
