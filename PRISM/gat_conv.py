from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        # Set the aggregation method to 'add' by default
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        # Initialize input/output channels and hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformation for source and destination nodes
        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src  # Tied weights for destination nodes

        # Attention parameters initialization
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))  # Source node attention
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))  # Destination node attention
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        # Initialize alpha values and attention weights
        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention=None):
        # Define the dimensions for heads and output channels
        H, C = self.heads, self.out_channels

        # Apply the linear transformation to the source and destination node features
        if isinstance(x, Tensor):
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        # Set x as a tuple for source and destination node features
        x = (x_src, x_dst)

        # If attention is not needed, return the mean of the source features
        if not attention:
            return x[0].mean(dim=1)

        # If tied attention is not provided, calculate attention scores
        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention  # Use provided tied attention scores

        # Add self-loops to the graph if required
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = min(x_src.size(0), x_dst.size(0) if x_dst is not None else x_src.size(0))
                edge_index, _ = remove_self_loops(edge_index)  # Remove self loops if present
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # Add self loops
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)  # Set diagonal for sparse matrix

        # Perform message passing and return the output
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        # Clear alpha to avoid leakage
        alpha = self._alpha
        self._alpha = None

        # Concatenate the output across heads or average them
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # If attention weights are required, return them along with the output
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        # Combine attention scores for source and destination nodes
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)  # Apply sigmoid activation to attention scores
        alpha = softmax(alpha, index, ptr, size_i)  # Apply softmax normalization

        # Store attention scores for debugging
        self._alpha = alpha

        # Dropout the attention scores for regularization
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Return the weighted sum of neighbor features
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        # Return a string representation of the class
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)