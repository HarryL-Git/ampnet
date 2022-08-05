import torch
import torch.nn as nn
import torch_geometric.nn
# from performer_pytorch import CrossAttention


class AMPConv(torch_geometric.nn.MessagePassing):
    def __init__(self, embed_dim, num_heads):
        super().__init__(aggr='mean')
        self.attn_output_weights = None
        self.attn_output = None
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=True)
        
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_i, x_j):
        if x_i.shape[1] % self.embed_dim != 0:
            print("Error, invalid configuration")

        x_i_reshape = torch.reshape(x_i, (x_i.shape[0], int(x_i.shape[1] / self.embed_dim), self.embed_dim))  # Shape becomes [num_edges, 1433, emb_dim]
        x_j_reshape = torch.reshape(x_j, (x_j.shape[0], int(x_j.shape[1] / self.embed_dim), self.embed_dim))

        # Forward through Multihead attention blocks
        self.attn_output, self.attn_output_weights = self.multi_head_attention(x_i_reshape, x_j_reshape, x_j_reshape)
        # x_j is source (target), K and V
        # x_i is destination, Q

        # Row index is feature index of destination

        output_reshape = torch.reshape(self.attn_output, (x_i.shape[0], x_i.shape[1]))

        return output_reshape
