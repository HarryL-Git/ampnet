import torch.nn.functional as F
import torch
from torch.nn import Linear, LayerNorm
from ..conv.amp_conv import AMPConv


class AMPNetClassifier(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, n_original_features, out_dim):
        super(AMPNetClassifier, self).__init__()

        self.conv1_embedding = None
        self.conv2_embedding = None
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.layer_norm = LayerNorm(
            n_original_features * embed_dim,
            elementwise_affine=False
        )

        self.conv1 = AMPConv(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.post_conv_linear1 = Linear(
            in_features=n_original_features * embed_dim,
            out_features=n_original_features * embed_dim
        )

        self.conv2 = AMPConv(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.post_conv_linear2 = Linear(
            in_features=n_original_features * embed_dim,
            out_features=n_original_features * embed_dim
        )

        self.linear_out = Linear(
            in_features=n_original_features * embed_dim,
            out_features=out_dim
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        self.conv1_embedding = x
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        self.conv2_embedding = x
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)
