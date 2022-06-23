import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

from project.utils.graph_utils import src_dot_dst, scaling, exp
from project.utils.deepinteract_constants import FEATURE_INDICES, RESIDUE_COUNT_LIMIT, NODE_COUNT_LIMIT



# based on https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py#L58

class MultiheadAttention(nn.Module):
    """Compute attention scores with a DGLGraph's node features."""

    def __init__(self, num_input_feats: int, num_output_feats: int,
                 num_heads: int, using_bias: bool):
        super().__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias

        # Define node features' query, key, and value tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

    def propagate_attention(self, graph: dgl.DGLGraph):
        
        # matmul K and Q
        graph.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        
        # Scale and clip attention scores
        graph.apply_edges(scaling('score', np.sqrt(self.num_output_feats), 5.0))

        # Apply softmax to attention scores, followed by clipping
        graph.apply_edges(exp('score', 5.0))

        # Send weighted values to target nodes
        e_ids = graph.edges()

        # V_h * score
        graph.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # z is denominator in softmax
        graph.send_and_recv(e_ids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))


    def forward(self, graph: dgl.DGLGraph, node_feats: torch.Tensor):
        
        with graph.local_scope():
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)

            # Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
            graph.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)

            # Disperse attention information
            self.propagate_attention(graph)

            # Compute final node and edge representations after multi-head attention
            # add 1e-6 to prevent division by zero
            h_out = graph.ndata['wV'] / (graph.ndata['z'] + torch.full_like(graph.ndata['z'], 1e-6))   

        # Return attention-updated node and edge representations
        return h_out

class GeometricTransformerModule(nn.Module):
    """A Geometric Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels: int,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            dropout_rate=0.1,
            knn=20,
            num_layers=4,
            feature_indices=FEATURE_INDICES,
            disable_geometric_mode=False
    ):
        super().__init__()

        """Geometry-Focused Graph Transformer Layer

        Parameters
        ----------
        shared_embed_size : int
            Size of shared embedding in a conformation module.
        dist_embed_size : int
            Size of distance embedding in a conformation module.
        dir_embed_size : int
            Size of direction embedding in a conformation module.
        orient_embed_size : int
            Size of orientation embedding in a conformation module.
        amide_embed_size : int
            Size of embedding in a conformation module for amide plane-amide plane normal vector angles.
        num_hidden_channels : int
            Hidden channel size for both nodes and edges.
        num_pre_res_blocks : int
            Number of residual blocks to apply prior to a residual reconnection.
        num_post_res_blocks : int
            Number of residual blocks to apply following a residual reconnection.
        activ_fn : Module
            Activation function to apply in MLPs.
        residual : bool
            Whether to use a residual update strategy for node features.
        num_attention_heads : int
            How many attention heads to apply to the input node features in parallel.
        norm_to_apply : str
            Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
        dropout_rate : float
            How much dropout (i.e. forget rate) to apply before activation functions.
        knn : int
            How many nearest neighbors were used when constructing node neighborhoods.
        num_layers : int
            How many layers of geometric attention to apply.
        feature_indices : dict
            A dictionary listing the start and end indices for each node and edge feature.
        disable_geometric_mode : bool
            Whether to convert the Geometric Transformer into the original Graph Transformer.
        """

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.knn = knn
        self.num_layers = num_layers
        self.feature_indices = feature_indices
        self.disable_geometric_mode = disable_geometric_mode

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels

        # Otherwise, default to using batch normalization
        self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiheadAttention(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])


    def run_gt_layer(self, graph: dgl.DGLGraph, node_feats: torch.Tensor):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection

        # Otherwise, default to using batch normalization
        node_feats = self.batch_norm1_node_feats(node_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out = self.mha_module(graph, node_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return node_feats

    def forward(self, graph: dgl.DGLGraph):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats = self.run_gt_layer(graph, graph.ndata['f'])
        return node_feats

