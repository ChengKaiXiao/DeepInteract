# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import argparse


# torch._C._jit_set_profiling_mode(False)
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class GaussianLayerNoEdge(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)




# class RBF(nn.Module):
#     def __init__(self, K, edge_types):
#         super().__init__()
#         self.K = K
#         self.means = nn.parameter.Parameter(torch.empty(K))
#         self.temps = nn.parameter.Parameter(torch.empty(K))
#         self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
#         self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
#         nn.init.uniform_(self.means, 0, 3)
#         nn.init.uniform_(self.temps, 0.1, 10)
#         nn.init.constant_(self.bias.weight, 0)
#         nn.init.constant_(self.mul.weight, 1)

#     def forward(self, x: Tensor, edge_types):
#         mul = self.mul(edge_types)
#         bias = self.bias(edge_types)
#         x = mul * x.unsqueeze(-1) + bias
#         mean = self.means.float()
#         temp = self.temps.float().abs()
#         return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


def get_arguments():
    parser = argparse.ArgumentParser(description="Architechture")
    parser.add_argument("--layers", type=int, default=4, metavar="L", help="num encoder layers")
    parser.add_argument("--blocks", type=int, default=1, metavar="L", help="num blocks")
    parser.add_argument("--embed-dim", type=int, default=128, metavar="H", help="encoder embedding dimension")
    parser.add_argument(
        "--ffn-embed-dim",
        type=int,
        default=128,
        metavar="F",
        help="encoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=16,
        metavar="A",
        help="num encoder attention heads",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, metavar="D", help="dropout probability"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.1,
        metavar="D",
        help="dropout probability for attention weights",
    )
    parser.add_argument(
        "--activation-dropout",
        type=float,
        default=0.0,
        metavar="D",
        help="dropout probability after activation in FFN",
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        metavar="D",
        help="dropout probability",
    )
    parser.add_argument(
        "--node-loss-weight",
        type=float,
        default=15,
        metavar="D",
        help="loss weight for node fitting",
    )
    parser.add_argument(
        "--min-node-loss-weight",
        type=float,
        default=1,
        metavar="D",
        help="loss weight for node fitting",
    )
    parser.add_argument(
        "--num-kernel",
        default=128,
        type=int,
    )

    # Original parameters
    # args.blocks = getattr(args, "blocks", 4)
    # args.layers = getattr(args, "layers", 12)
    # args.embed_dim = getattr(args, "embed_dim", 768)
    # args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
    # args.attention_heads = getattr(args, "attention_heads", 48)
    # args.input_dropout = getattr(args, "input_dropout", 0.0)
    # args.dropout = getattr(args, "dropout", 0.1)
    # args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    # args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    # args.node_loss_weight = getattr(args, "node_loss_weight", 15)
    # args.min_node_loss_weight = getattr(args, "min_node_loss_weight", 1)
    # args.eng_loss_weight = getattr(args, "eng_loss_weight", 1)
    # args.num_kernel = getattr(args, "num_kernel", 128)
    # args = parser.parse_args()

    args = parser.parse_args()

    return args

class Graphormer3Dmodule(nn.Module):
    """
    Input 
    node_features: graph.ndata['f']
    pos: graph.ndata['x']
    """

    def __init__(self, args):
        super().__init__()
        # self.args = get_arguments()
        self.args = args
        self.edge_types = 20 * 20 # 20 ammino acids

        # embed atom types
        self.atom_types = 20
        self.input_dropout = self.args.input_dropout    # dropout prob
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    num_attention_heads=self.args.attention_heads,
                    dropout=self.args.dropout,
                    attention_dropout=self.args.attention_dropout,
                    activation_dropout=self.args.activation_dropout,
                )
                for _ in range(self.args.layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.args.embed_dim)

        K = self.args.num_kernel

        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayerNoEdge(K)
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.args.attention_heads
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.args.embed_dim)


    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        return super().set_num_updates(num_updates)

    def forward(self, node_feats: Tensor, pos: Tensor):
        n_node = node_feats.size()[0]

        #node_feats = self.input_embedding(node_feats)

        node_feats = node_feats.unsqueeze(dim=0) # add 1 batch dimension 
        pos = pos.unsqueeze(dim=0) # add 1 batch dimension 

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)

        gbf_feature = self.gbf(dist)

        # Centrality encoding
        graph_node_feature = (
            node_feats
            + self.edge_proj(gbf_feature.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(graph_node_feature, p=self.input_dropout, training=self.training)
        output = output.transpose(0, 1).contiguous()

        # Spatial encoding
        graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.args.blocks):
            for enc_layer in self.layers:   # loop over all encoder layers
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        
        output = output.transpose(0, 1)

        return output.squeeze(0)


# test = GaussianLayerNoEdge()
# arg = get_arguments()
# model = Graphormer3Dmodule(arg)
