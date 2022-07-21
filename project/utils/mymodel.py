import os
from argparse import ArgumentParser
from math import sqrt

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import wandb
from dgl.nn.pytorch import GraphConv
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.utils.deepinteract_constants import FEATURE_INDICES, RESIDUE_COUNT_LIMIT, NODE_COUNT_LIMIT
from project.utils.deepinteract_utils import construct_interact_tensor, glorot_orthogonal, get_geo_feats_from_edges, \
    construct_subsequenced_interact_tensors, insert_interact_tensor_logits, \
    remove_padding, remove_subsequenced_input_padding, calculate_top_k_prec, calculate_top_k_recall, extract_object
from project.utils.vision_modules import DeepLabV3Plus

from project.utils.graphormer3D import Graphormer3Dmodule, get_arguments

class GraphormerModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_feats):
        # print(node_feats.size())
        return node_feats

class ResBlock(nn.Module):
    """A residual block for a conformation module."""

    def __init__(self, hidden_channels, activ_fn=nn.SiLU(), norm_to_apply='batch'):
        super().__init__()

        # Record parameters given
        self.activ_fn = activ_fn

        # Define projection layers for a conformation module residual block
        norm_layer = nn.LayerNorm(hidden_channels) if norm_to_apply == 'layer' else nn.BatchNorm1d(hidden_channels)
        self.res_block = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn,
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn,
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        for layer in self.res_block:
            not_norm_layer = not (hasattr(layer, 'normalized_shape') or hasattr(layer, 'running_mean'))
            if hasattr(layer, 'weight') and not_norm_layer:  # Skip init for activation functions and normalizing layers
                glorot_orthogonal(layer.weight, scale=scale)
                layer.bias.data.fill_(0)

    def forward(self, x):
        """Perform a forward pass using residual layers with intermediate activation functions applied."""
        x_res = x
        for layer in self.res_block:
            x_res = layer(x_res)
        return x + x_res


class SEBlock(torch.nn.Module):
    """A squeeze-and-excitation block for PyTorch."""

    def __init__(self, ch, ratio=16):
        super(SEBlock, self).__init__()
        self.ratio = ratio
        self.linear1 = torch.nn.Linear(ch, ch // ratio)
        self.linear2 = torch.nn.Linear(ch // ratio, ch)
        self.act = nn.ReLU()

    def forward(self, in_block):
        x = torch.reshape(in_block, (in_block.shape[0], in_block.shape[1], -1))
        x = torch.mean(x, dim=-1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = torch.sigmoid(x)
        return torch.einsum('bcij,bc->bcij', in_block, x)


class ResNet(nn.Module):
    """A custom ResNet module for PyTorch."""

    # Parameter initialization
    def __init__(self,
                 num_channels,
                 num_chunks,
                 module_name,
                 activ_fn=F.elu,
                 inorm=False,
                 initial_projection=False,
                 extra_blocks=False,
                 dilation_cycle=None,
                 verbose=False):

        self.num_channel = num_channels
        self.num_chunks = num_chunks
        self.module_name = module_name
        self.activ_fn = activ_fn
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = [1, 2, 4, 8] if dilation_cycle is None else dilation_cycle
        self.verbose = verbose

        super(ResNet, self).__init__()

        if self.initial_projection:
            self.add_module(f'resnet_{self.module_name}_init_proj', nn.Conv2d(in_channels=num_channels,
                                                                              out_channels=num_channels,
                                                                              kernel_size=(1, 1)))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=dilation_rate,
                                          padding=dilation_rate))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block',
                                SEBlock(num_channels, ratio=16))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=(1, 1),
                                          padding=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_se_block',
                                SEBlock(num_channels, ratio=16))

    def forward(self, x):
        """Compute ResNet output."""
        activ_fn = self.activ_fn

        if self.initial_projection:
            x = self._modules[f'resnet_{self.module_name}_init_proj'](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block'](x)

                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_extra{i}_se_block'](x)

                x = x + _residual

        return x


class MultiHeadRegionalAttention(nn.Module):
    """A multi-head attention block for PyTorch that operates regionally."""

    @staticmethod
    def get_stretch_weight(s):
        w = np.zeros((s * s, 1, 1, s, s))
        for i in range(s):
            for j in range(s):
                w[s * i + j, 0, 0, i, j] = 1
        return np.asarray(w).astype(np.float32)

    def __init__(self, in_dim=3, region_size=3, d_k=16, d_v=32, n_head=4, att_drop=0.1, output_score=False):
        super(MultiHeadRegionalAttention, self).__init__()
        self.temper = int(np.sqrt(d_k))
        self.dk_per_head = d_k // n_head
        self.dv_per_head = d_v // n_head
        self.dropout_layer = nn.Dropout(att_drop)
        self.output_score = output_score
        self.q_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.k_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.v_layer = nn.Conv2d(in_dim, d_v, kernel_size=(1, 1), bias=False)
        self.softmax_layer = nn.Softmax(1)
        self.stretch_layer = nn.Conv3d(in_channels=1,
                                       out_channels=region_size * region_size,
                                       kernel_size=(1, region_size, region_size),
                                       bias=False,
                                       padding=(0, 1, 1))
        self.stretch_layer.weight = nn.Parameter(
            torch.tensor(self.get_stretch_weight(region_size)), requires_grad=False
        )

    def forward(self, x):
        """Compute attention output and attention score."""
        Q = self.stretch_layer(self.q_layer(x).unsqueeze(1))
        K = self.stretch_layer(self.k_layer(x).unsqueeze(1))
        V = self.stretch_layer(self.v_layer(x).unsqueeze(1))
        qk = torch.mul(Q, K).permute(0, 2, 1, 3, 4)
        qk1 = qk.view((-1, self.dk_per_head, qk.shape[2], qk.shape[3], qk.shape[4]))
        attention_score = self.softmax_layer(torch.div(torch.sum(qk1, 1), self.temper))
        attention_score2 = self.dropout_layer(attention_score)
        attention_score2 = torch.repeat_interleave(attention_score2.unsqueeze(0).permute(0, 2, 1, 3, 4),
                                                   repeats=self.dv_per_head, dim=2)
        attention_out = torch.sum(torch.mul(attention_score2, V), dim=1)
        return attention_out, attention_score if self.output_score else attention_out


class ResNet2DInputWithOptAttention(nn.Module):
    """A ResNet and (optionally) regionally-attentive convolution module for a pair of 2D feature tensors."""

    def __init__(self,
                 num_chunks=4,
                 init_channels=128,
                 num_channels=128,
                 num_classes=2,
                 use_attention=False,
                 n_head=4,
                 module_name=None,
                 activ_fn=F.elu,
                 dropout=0.1,
                 verbose=False):
        super(ResNet2DInputWithOptAttention, self).__init__()
        self.num_chunks = num_chunks
        self.init_channels = init_channels
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.n_head = n_head
        self.module_name = module_name
        self.activ_fun = activ_fn
        self.dropout = dropout
        self.verbose = verbose

        self.add_module('conv2d_1', nn.Conv2d(in_channels=self.init_channels,
                                              out_channels=self.num_channels,
                                              kernel_size=(1, 1),
                                              padding=(0, 0)))
        self.add_module('inorm_1', nn.InstanceNorm2d(self.num_channels, eps=1e-06, affine=True))

        self.add_module('base_resnet', ResNet(num_channels,
                                              self.num_chunks,
                                              module_name='base_resnet',
                                              activ_fn=self.activ_fun,
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))

        self.add_module('phase2_resnet', ResNet(num_channels,
                                                num_chunks=1,
                                                module_name='bin_resnet',
                                                activ_fn=self.activ_fun,
                                                inorm=False,
                                                initial_projection=True,
                                                extra_blocks=True))
        self.add_module('phase2_conv', nn.Conv2d(in_channels=self.num_channels,
                                                 out_channels=self.num_classes,
                                                 kernel_size=(1, 1),
                                                 padding=(0, 0)))
        if self.use_attention:
            self.add_module('MHA2D_1', MultiHeadRegionalAttention(self.num_channels,
                                                                  d_v=self.num_channels,
                                                                  n_head=self.n_head,
                                                                  att_drop=self.dropout,
                                                                  output_score=True))
            self.add_module('MHA2D_2', MultiHeadRegionalAttention(self.num_channels,
                                                                  d_v=self.num_channels,
                                                                  n_head=self.n_head,
                                                                  att_drop=self.dropout,
                                                                  output_score=True))

        # Reset learnable parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # Reinitialize final output layer
        final_layer_bias = self._modules['phase2_conv'].bias.clone()
        final_layer_bias[1] = -7.0  # -7 chosen as the second term's bias s.t. positives are predicted with prob=0.001
        self._modules['phase2_conv'].bias = nn.Parameter(final_layer_bias, requires_grad=True)

    def forward(self, f2d_tile: torch.Tensor):
        """Compute final convolution output."""
        activ_fun = self.activ_fun
        out_conv2d_1 = self._modules['conv2d_1'](f2d_tile)
        out_inorm_1 = activ_fun(self._modules['inorm_1'](out_conv2d_1))

        # First ResNet
        out_base_resnet = activ_fun(self._modules['base_resnet'](out_inorm_1))
        if self.use_attention:
            out_base_resnet, attention_scores_1 = self._modules['MHA2D_1'](out_base_resnet)
            out_base_resnet = activ_fun(out_base_resnet)

        # Second ResNet
        out_bin_predictor = activ_fun(self._modules['phase2_resnet'](out_base_resnet))
        if self.use_attention:
            out_bin_predictor, attention_scores_2 = self._modules['MHA2D_2'](out_bin_predictor)
            out_bin_predictor = activ_fun(out_bin_predictor)

        # Output convolution
        out_layer = self._modules['phase2_conv'](out_bin_predictor)
        return out_layer



# ------------------
# Lightning Modules
# ------------------

class LitGINI(pl.LightningModule):
    """A geometry-focused inter-graph node interaction (GINI) module."""

    def __init__(self, num_node_input_feats: int, num_edge_input_feats: int, gnn_activ_fn=nn.SiLU(),
                 num_classes=2, max_num_graph_nodes=NODE_COUNT_LIMIT, max_num_residues=RESIDUE_COUNT_LIMIT,
                 testing_with_casp_capri=False, training_with_db5=False, pos_prob_threshold=0.5, 
                 num_gnn_hidden_channels=512,
                 num_gnn_attention_heads=32, knn=20, interact_module_type='dil_resnet', num_interact_layers=14,
                 num_interact_hidden_channels=128, use_interact_attention=False, num_interact_attention_heads=4,
                 num_epochs=50, pn_ratio=0.1, dropout_rate=0.2, metric_to_track='val_ce',
                 weight_decay=1e-2, batch_size=1, lr=1e-3, pad=False, use_wandb_logger=True,
                 weight_classes=False, fine_tune=False, ckpt_path=None):
        """Initialize all the parameters for a LitGINI module."""
        super().__init__()

        # Build the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.gnn_activ_fn = gnn_activ_fn
        self.num_classes = num_classes
        self.max_num_graph_nodes = max_num_graph_nodes
        self.max_num_residues = max_num_residues
        self.testing_with_casp_capri = testing_with_casp_capri
        self.training_with_db5 = training_with_db5
        self.pos_prob_threshold = pos_prob_threshold

        # GNN module's keyword arguments provided via the command line
        self.num_gnn_hidden_channels = num_gnn_hidden_channels
        self.num_gnn_attention_heads = num_gnn_attention_heads
        self.nbrhd_size = knn

        # Interaction module's keyword arguments provided via the command line
        self.interact_module_type = interact_module_type
        self.num_interact_layers = num_interact_layers
        self.num_interact_hidden_channels = num_interact_hidden_channels
        self.use_interact_attention = use_interact_attention
        self.num_interact_attention_heads = num_interact_attention_heads

        # Derive shortcut booleans for convenient future reference
        self.using_dil_resnet = self.interact_module_type.lower() == 'dil_resnet'
        self.using_deeplab = self.interact_module_type.lower() == 'deeplab'

        # Model hyperparameter keyword arguments provided via the command line
        self.num_epochs = num_epochs
        self.pn_ratio = pn_ratio
        self.dropout_rate = dropout_rate
        self.metric_to_track = metric_to_track
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.pad = pad
        self.use_wandb_logger = use_wandb_logger  # Whether to use WandB as the primary means of logging
        self.weight_classes = weight_classes  # Whether to use class weighting in our training Cross Entropy
        self.fine_tune = fine_tune  # Whether to fine-tune a trained LitGINI on a new dataset
        self.ckpt_path = ckpt_path  # A path to a trained LitGINI checkpoint given if fine-tuning

        # Set up GNN node and edge embedding layers
        self.node_embedding = nn.Linear(self.num_node_input_feats, self.num_gnn_hidden_channels, bias=False)
        self.embed_reduction = nn.Linear(self.num_gnn_hidden_channels, self.num_interact_hidden_channels, bias=False)

        # Assemble the layers of the network
        if self.fine_tune:
            # Load in trained LitGINI
            lit_gini = LitGINI.load_from_checkpoint(self.ckpt_path,
                                                    use_wandb_logger=use_wandb_logger,
                                                    batch_size=self.batch_size,
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay,
                                                    dropout_rate=self.dropout_rate)
            self.gnn_module, self.interact_module = lit_gini.gnn_module, lit_gini.interact_module
            # Freeze the interaction module during fine-tuning
            for param in self.interact_module.parameters():
                param.requires_grad = False
        else:
            self.build_gnn_module(), self.build_interaction_module()

        # Declare loss functions and metrics for training, validation, and testing
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.train_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.train_recall = tm.Recall(num_classes=self.num_classes, average=None)

        self.val_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.val_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.val_recall = tm.Recall(num_classes=self.num_classes, average=None)
        self.val_auroc = tm.AUROC(num_classes=self.num_classes, average=None)
        self.val_auprc = tm.AveragePrecision(num_classes=self.num_classes)
        self.val_f1 = tm.F1(num_classes=self.num_classes, average=None)

        self.test_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.test_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.test_recall = tm.Recall(num_classes=self.num_classes, average=None)
        self.test_auroc = tm.AUROC(num_classes=self.num_classes, average=None)
        self.test_auprc = tm.AveragePrecision(num_classes=self.num_classes)
        self.test_f1 = tm.F1(num_classes=self.num_classes, average=None)

        # Reset learnable parameters and log hyperparameters
        self.save_hyperparameters(ignore=['gnn_activ_fn'])

    def build_gnn_module(self):
        """Define all layers for the chosen GNN module."""
        # gnn_layers = [nn.Identity()]
        # gnn_layers = [GraphormerModule()]
        arguments = get_arguments()
        gnn_layers = [Graphormer3Dmodule(arguments)]
        self.gnn_module = nn.ModuleList(gnn_layers)

    def get_interact_module(self):
        """Retrieve an interaction module of a specific type (e.g. Dilated ResNet or DeepLabV3Plus)."""
        if self.using_deeplab:
            interact_module = DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_depth=self.num_interact_layers,
                encoder_output_stride=16,
                decoder_channels=self.num_interact_hidden_channels,
                decoder_atrous_rates=(12, 24, 36),
                in_channels=self.num_gnn_hidden_channels * 2,
                classes=self.num_classes,
                upsampling=4
            )
        else:  # Otherwise, default to using our dilated ResNet with squeeze-and-excitation (SE)
            interact_module = ResNet2DInputWithOptAttention(num_chunks=self.num_interact_layers,
                                                            init_channels=self.num_gnn_hidden_channels * 2,
                                                            num_channels=self.num_interact_hidden_channels,
                                                            num_classes=self.num_classes,
                                                            use_attention=self.use_interact_attention,
                                                            n_head=self.num_interact_attention_heads,
                                                            activ_fn=F.elu,
                                                            dropout=self.dropout_rate,
                                                            verbose=False)
        return interact_module

    def build_interaction_module(self):
        """Define all layers for the chosen interaction module."""
        # Dilated ResNets and DeepLabV3Plus package all their forward pass logic
        self.interact_module = self.get_interact_module()

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph: dgl.DGLGraph):
        """Make a forward pass through a single GNN module."""
        # Embed input features a priori
        node_feats = self.node_embedding(graph.ndata['f']).squeeze()

        for layer in self.gnn_module:
            node_feats = layer(node_feats, graph.ndata['x'])  # Geometric Transformers can handle their own depth

        # reduce embedding dimensions
        node_feats = self.embed_reduction(node_feats)

        return [node_feats]

    def interact_forward(self, interact_tensor: torch.Tensor):
        """Make a forward pass through the interaction module."""
        # Dilated ResNets and DeepLabV3Plus package all their forward pass logic
        logits = self.interact_module(interact_tensor)
        return logits

    def shared_step(self, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph, return_representations=False):
        """Make a forward pass through the entire siamese network."""
        # Learn structural features for each structure's nodes
        graph1_node_feats = self.gnn_forward(graph1)
        graph2_node_feats = self.gnn_forward(graph2)

        # Interleave node features from both graphs to achieve the desired interaction tensor
        current_phase_batch_size = len(graph1_node_feats)  # Use feature tensor collection length as phase's batch size
        graph1_is_within_size_limit = graph1.num_nodes() < self.max_num_residues
        graph2_is_within_size_limit = graph2.num_nodes() < self.max_num_residues
        both_graphs_within_limit = graph1_is_within_size_limit and graph2_is_within_size_limit
        # high_mem_model = self.using_dil_resnet  # Reinstate if wanting to restrict memory consumption with dil_resnet
        high_mem_model = False  # Ignore subsequencing for Dilated ResNets
        subsequencing_input = both_graphs_within_limit is False and current_phase_batch_size == 1 and high_mem_model
        if subsequencing_input:  # Need to subsequence input complexes to avoid running out of GPU memory
            interact_tensors = construct_subsequenced_interact_tensors(
                graph1_node_feats, graph2_node_feats, current_phase_batch_size, pad=self.pad,
                max_len=self.max_num_residues
            )
        else:
            interact_tensors = [
                construct_interact_tensor(
                    g1_node_feats, g2_node_feats, pad=self.pad, max_len=self.max_num_residues
                )
                for g1_node_feats, g2_node_feats in zip(graph1_node_feats, graph2_node_feats)
            ]

        # Handle for optional padding
        if self.pad:  # When subsequencing, we must address padding inside insert() below
            interact_tensors = torch.cat(interact_tensors)
            # Predict node-node pair interactions using an interaction module (i.e. series of interaction layers)
            logits = self.interact_forward(interact_tensors)
            # Remove any added padding from learned interaction tensors
            remove_padding_fn = remove_subsequenced_input_padding if subsequencing_input else remove_padding
            logits_list = remove_padding_fn(logits, graph1_node_feats, graph2_node_feats, self.max_num_residues)
        else:
            logits_list = [self.interact_forward(interact_tensor) for interact_tensor in interact_tensors]

        # Recombine subsequenced logits into the original interaction tensor's shape
        if subsequencing_input:
            if current_phase_batch_size == 1:
                interact_tensor = torch.zeros(1,
                                              self.num_classes,
                                              graph1.num_nodes(),
                                              graph2.num_nodes(),
                                              device=self.device)
                interact_tensor = insert_interact_tensor_logits(logits_list, interact_tensor, self.max_num_residues)
                logits_list = [interact_tensor]
            else:
                # TODO: Implement subsequence batching of graph batches
                raise NotImplementedError

        if return_representations:
            # Return network prediction and learned node and edge representations for both graphs
            g1_nf, g1_ef = graph1.ndata['f'].detach().cpu().numpy(), graph1.edata['f'].detach().cpu().numpy()
            g2_nf, g2_ef = graph2.ndata['f'].detach().cpu().numpy(), graph2.edata['f'].detach().cpu().numpy()
            return logits_list, g1_nf, g1_ef, g2_nf, g2_ef
        else:
            return logits_list

    def downsample_examples(self, examples: torch.tensor):
        """Randomly sample enough negative pairs to achieve requested positive-negative class ratio (via shuffling)."""
        examples = examples[torch.randperm(len(examples))]  # Randomly shuffle training examples
        pos_examples = examples[examples[:, 2] == 1]  # Find out how many interacting pairs there are
        num_neg_pairs_to_sample = int(len(pos_examples) / self.pn_ratio)  # Determine negative sample size
        neg_examples = examples[examples[:, 2] == 0][:num_neg_pairs_to_sample]  # Sample negative pairs
        downsampled_examples = torch.cat((neg_examples, pos_examples))
        return downsampled_examples

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Separate training batch from validation batch (with the latter being used for visualizations)
        # train_batch, val_batch = batch['train_batch'], batch['val_batch']
        train_batch = batch['train_batch']
        # Make a forward pass through the network for a singular training batch of protein complexes
        graph1, graph2, examples_list, filepaths = train_batch[0], train_batch[1], train_batch[2], train_batch[3]

        # Forward propagate with network layers
        logits_list = self.shared_step(graph1, graph2)  # The forward method must be named something new

        # Collect flattened sampled logits and their corresponding labels
        sampled_examples = []
        sampled_logits = torch.tensor([], device=self.device)
        for logits, examples in zip(logits_list, examples_list):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            # examples = self.downsample_examples(examples) if self.pn_ratio > 0.0015 else examples
            sampled_examples.append(examples)  # Add modified examples into new list
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(sampled_examples)

        # Down-weight negative pairs to achieve desired PN weight, and then up-weight positive pairs appropriately
        if self.weight_classes:
            neg_class_weight = 1.0  # Modify by how much negative class samples are weighted
            pos_class_weight = 5.0  # Modify by how much positive class samples are weighted
            class_weights = torch.tensor([neg_class_weight, pos_class_weight], device=self.device)
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights  # Weight each class separately for a given complex
            )
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate the protein interface prediction (PICP) loss along with additional PICP metrics
        loss = loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        train_acc = self.train_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        train_prec = self.train_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        train_recall = self.train_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex

        # Log training step metric(s)
        self.log(f'train_ce', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        return {
            'loss': loss,
            'train_acc': train_acc,
            'train_prec': train_prec,
            'train_recall': train_recall
        }

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        """Lightning calls this at the end of every training epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        train_accs = torch.cat([output_dict['train_acc'].unsqueeze(0) for output_dict in outputs])
        train_precs = torch.cat([output_dict['train_prec'].unsqueeze(0) for output_dict in outputs])
        train_recalls = torch.cat([output_dict['train_recall'].unsqueeze(0) for output_dict in outputs])

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        train_accs = torch.cat([train_acc for train_acc in self.all_gather(train_accs)], dim=0)
        train_precs = torch.cat([train_prec for train_prec in self.all_gather(train_precs)], dim=0)
        train_recalls = torch.cat([train_recall for train_recall in self.all_gather(train_recalls)], dim=0)

        # Reset training TorchMetrics for all devices
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_recall.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_train_acc', torch.median(train_accs), batch_size=self.batch_size)  # Log MedAccuracy of an epoch
        self.log('med_train_prec', torch.median(train_precs), batch_size=self.batch_size)  # Log MedPrecision of an epoch
        self.log('med_train_recall', torch.median(train_recalls), batch_size=self.batch_size)  # Log MedRecall of an epoch

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        # Make a forward pass through the network for a singular validation batch of protein complexes
        graph1, graph2, examples_list, filepaths = batch[0], batch[1], batch[2], batch[3]

        # Forward propagate with network layers
        logits_list = self.shared_step(graph1, graph2)

        # Collect flattened, sampled logits and their corresponding labels
        sampled_logits = torch.tensor([], device=self.device)
        for i, (logits, examples) in enumerate(zip(logits_list, examples_list)):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            examples_list[i] = examples  # Replace original examples tensor in-place
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(examples_list)

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate top-k metrics
        calculating_l_by_n_metrics = True
        # Log only first 50 validation top-k metrics to limit algorithmic complexity due to sorting (if requested)
        # calculating_l_by_n_metrics = batch_idx in [i for i in range(50)]
        if calculating_l_by_n_metrics:
            l = graph1.num_nodes() + graph2.num_nodes()
            sorted_pred_indices = torch.argsort(preds[:, 1], descending=True)
            top_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=10)
            top_l_by_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 10))
            top_l_by_5_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 5))
            top_l_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=l)
            top_l_by_2_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=(l // 2))
            top_l_by_5_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=(l // 5))

        # Calculate the protein interface prediction (PICP) loss along with additional PIP metrics
        loss = self.loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        val_acc = self.val_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        val_prec = self.val_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        val_recall = self.val_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex
        val_f1 = self.val_f1(preds_rounded, labels)[1]  # Calculate F1 score of a single complex
        val_auroc = self.val_auroc(preds, labels)[1]  # Calculate AUROC of a complex
        val_auprc = self.val_auprc(preds, labels)[1]  # Calculate AveragePrecision (i.e. AUPRC) of a complex

        # Log validation step metric(s)
        self.log(f'val_ce', loss, sync_dist=True, batch_size=self.batch_size)
        if calculating_l_by_n_metrics:
            self.log('val_top_10_prec', top_10_prec, sync_dist=True, batch_size=self.batch_size)
            self.log('val_top_l_by_10_prec', top_l_by_10_prec, sync_dist=True, batch_size=self.batch_size)
            self.log('val_top_l_by_5_prec', top_l_by_5_prec, sync_dist=True, batch_size=self.batch_size)
            self.log('val_top_l_recall', top_l_recall, sync_dist=True, batch_size=self.batch_size)
            self.log('val_top_l_by_2_recall', top_l_by_2_recall, sync_dist=True, batch_size=self.batch_size)
            self.log('val_top_l_by_5_recall', top_l_by_5_recall, sync_dist=True, batch_size=self.batch_size)

        return {
            'loss': loss,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auroc': val_auroc,
            'val_auprc': val_auprc
        }

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        """Lightning calls this at the end of every validation epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        val_accs = torch.cat([output_dict['val_acc'].unsqueeze(0) for output_dict in outputs])
        val_precs = torch.cat([output_dict['val_prec'].unsqueeze(0) for output_dict in outputs])
        val_recalls = torch.cat([output_dict['val_recall'].unsqueeze(0) for output_dict in outputs])
        val_f1s = torch.cat([output_dict['val_f1'].unsqueeze(0) for output_dict in outputs])
        val_aurocs = torch.cat([output_dict['val_auroc'].unsqueeze(0) for output_dict in outputs])
        val_auprcs = torch.cat([output_dict['val_auprc'].unsqueeze(0) for output_dict in outputs])

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        val_accs = torch.cat([val_acc for val_acc in self.all_gather(val_accs)])
        val_precs = torch.cat([val_prec for val_prec in self.all_gather(val_precs)])
        val_recalls = torch.cat([val_recall for val_recall in self.all_gather(val_recalls)])
        val_f1s = torch.cat([val_f1 for val_f1 in self.all_gather(val_f1s)])
        val_aurocs = torch.cat([val_auroc for val_auroc in self.all_gather(val_aurocs)])
        val_auprcs = torch.cat([val_auprc for val_auprc in self.all_gather(val_auprcs)])

        # Reset validation TorchMetrics for all devices
        self.val_acc.reset()
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_val_acc', torch.median(val_accs), batch_size=self.batch_size)  # Log MedAccuracy of an epoch
        self.log('med_val_prec', torch.median(val_precs), batch_size=self.batch_size)  # Log MedPrecision of an epoch
        self.log('med_val_recall', torch.median(val_recalls), batch_size=self.batch_size)  # Log MedRecall of an epoch
        self.log('med_val_f1', torch.median(val_f1s), batch_size=self.batch_size)  # Log MedF1 of an epoch
        self.log('med_val_auroc', torch.median(val_aurocs))  # Log MedAUROC of an epoch
        self.log('med_val_auprc', torch.median(val_auprcs))  # Log epoch MedAveragePrecision

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        # Make a forward pass through the network for a singular test batch of protein complexes
        graph1, graph2, examples_list, filepaths = batch[0], batch[1], batch[2], batch[3]

        # Forward propagate with network layers
        logits_list = self.shared_step(graph1, graph2)

        # Collect flattened, sampled logits and their corresponding labels
        sampled_logits = torch.tensor([], device=self.device)
        for i, (logits, examples) in enumerate(zip(logits_list, examples_list)):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            examples_list[i] = examples  # Replace original examples tensor in-place
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(examples_list)

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate top-k metrics
        l = min(graph1.num_nodes(), graph2.num_nodes())  # Use the smallest length of the two chains as our denominator
        sorted_pred_indices = torch.argsort(preds[:, 1], descending=True)
        top_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=10)
        top_l_by_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 10))
        top_l_by_5_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 5))
        top_l_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=l)
        top_l_by_2_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=(l // 2))
        top_l_by_5_recall = calculate_top_k_recall(sorted_pred_indices, labels, k=(l // 5))

        # Calculate the protein interface prediction (PICP) loss along with additional PIP metrics
        loss = self.loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        test_acc = self.test_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        test_prec = self.test_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        test_recall = self.test_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex
        test_f1 = self.test_f1(preds_rounded, labels)[1]  # Calculate F1 score of a single complex
        test_auroc = self.test_auroc(preds, labels)[1]  # Calculate AUROC of a complex
        test_auprc = self.test_auprc(preds, labels)[1]  # Calculate AveragePrecision (i.e. AUPRC) of a complex

        # Manually evaluate test performance by collecting all predicted and ground-truth interaction tensors
        test_preds = preds.detach()
        test_preds_rounded = test_preds.clone()
        test_preds_rounded[:, 0] = (test_preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        test_preds_rounded[:, 1] = (test_preds[:, 1] >= self.pos_prob_threshold).float()
        test_preds = test_preds[:, 1].reshape(graph1.num_nodes(), graph2.num_nodes()).cpu().numpy()
        test_preds_rounded = test_preds_rounded[:, 1].reshape(graph1.num_nodes(), graph2.num_nodes()).cpu().numpy()

        test_labels = examples[:, 2].detach()
        test_labels = test_labels.reshape(graph1.num_nodes(), graph2.num_nodes()).float().cpu().numpy()

        # Log test step metric(s)
        self.log(f'test_ce', loss, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_10_prec', top_10_prec, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_l_by_10_prec', top_l_by_10_prec, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_l_by_5_prec', top_l_by_5_prec, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_l_recall', top_l_recall, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_l_by_2_recall', top_l_by_2_recall, sync_dist=True, batch_size=self.batch_size)
        self.log('test_top_l_by_5_recall', top_l_by_5_recall, sync_dist=True, batch_size=self.batch_size)

        return {
            'loss': loss,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auroc': test_auroc,
            'test_auprc': test_auprc,
            'test_preds': test_preds,
            'test_preds_rounded': test_preds_rounded,
            'test_labels': test_labels,
            'top_10_prec': top_10_prec,
            'top_l_by_10_prec': top_l_by_10_prec,
            'top_l_by_5_prec': top_l_by_5_prec,
            'top_l_recall': top_l_recall,
            'top_l_by_2_recall': top_l_by_2_recall,
            'top_l_by_5_recall': top_l_by_5_recall,
            'target': filepaths[0].split(os.sep)[-1][:4]
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT):
        """Lightning calls this at the end of every test epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        test_accs = torch.cat([output_dict['test_acc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_precs = torch.cat([output_dict['test_prec'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_recalls = torch.cat([output_dict['test_recall'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_f1s = torch.cat([output_dict['test_f1'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_aurocs = torch.cat([output_dict['test_auroc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_auprcs = torch.cat([output_dict['test_auprc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        test_accs = torch.cat([test_acc for test_acc in self.all_gather(test_accs)])
        test_precs = torch.cat([test_prec for test_prec in self.all_gather(test_precs)])
        test_recalls = torch.cat([test_recall for test_recall in self.all_gather(test_recalls)])
        test_f1s = torch.cat([test_f1 for test_f1 in self.all_gather(test_f1s)])
        test_aurocs = torch.cat([test_auroc for test_auroc in self.all_gather(test_aurocs)])
        test_auprcs = torch.cat([test_auprc for test_auprc in self.all_gather(test_auprcs)])

        if self.use_wandb_logger:
            test_preds = [wandb.Image(output_dict['test_preds']) for output_dict in outputs]  # Convert to image
            test_preds_rounded = [wandb.Image(output_dict['test_preds_rounded']) for output_dict in outputs]  # Rounded
            test_labels = [wandb.Image(output_dict['test_labels']) for output_dict in outputs]  # Convert to image
        else:  # Assume we are instead using the TensorBoardLogger
            test_preds = [output_dict['test_preds'] for output_dict in outputs]
            test_preds_rounded = [(output_dict['test_preds_rounded']) for output_dict in outputs]
            test_labels = [output_dict['test_labels'] for output_dict in outputs]

        # Write out test top-k metric results to CSV
        metrics_data = {
            'top_10_prec': [extract_object(output_dict['top_10_prec']) for output_dict in outputs],
            'top_l_by_10_prec': [extract_object(output_dict['top_l_by_10_prec']) for output_dict in outputs],
            'top_l_by_5_prec': [extract_object(output_dict['top_l_by_5_prec']) for output_dict in outputs],
            'top_l_recall': [extract_object(output_dict['top_l_recall']) for output_dict in outputs],
            'top_l_by_2_recall': [extract_object(output_dict['top_l_by_2_recall']) for output_dict in outputs],
            'top_l_by_5_recall': [extract_object(output_dict['top_l_by_5_recall']) for output_dict in outputs],
            'target': [extract_object(output_dict['target']) for output_dict in outputs],
        }
        metrics_df = pd.DataFrame(data=metrics_data)
        metrics_df_name_prefix = 'dips_plus_test'
        metrics_df_name_prefix = 'casp_capri' if self.testing_with_casp_capri else metrics_df_name_prefix
        metrics_df_name_prefix = 'db5_plus_test' if self.training_with_db5 else metrics_df_name_prefix
        metrics_df_name = metrics_df_name_prefix + '_top_metrics.csv'
        metrics_df.to_csv(metrics_df_name)

        if not self.testing_with_casp_capri:  # Testing with either DIPS-Plus or DB5-Plus
            # Filter out all but the first 55 test predictions and labels to reduce storage requirements
            test_preds, test_preds_rounded, test_labels = test_preds[:55], test_preds_rounded[:55], test_labels[:55]

        # Reset test TorchMetrics for all devices
        self.test_acc.reset()
        self.test_prec.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_test_acc', torch.median(test_accs), batch_size=self.batch_size)  # Log MedAccuracy of an epoch
        self.log('med_test_prec', torch.median(test_precs), batch_size=self.batch_size)  # Log MedPrecision of an epoch
        self.log('med_test_recall', torch.median(test_recalls), batch_size=self.batch_size)  # Log MedRecall of an epoch
        self.log('med_test_f1', torch.median(test_f1s), batch_size=self.batch_size)  # Log MedF1 of an epoch
        self.log('med_test_auroc', torch.median(test_aurocs))  # Log MedAUROC of an epoch
        self.log('med_test_auprc', torch.median(test_auprcs))  # Log epoch MedAveragePrecision

        # Log test predictions with their ground-truth interaction tensors to WandB for visual inspection
        if self.use_wandb_logger:
            self.trainer.logger.experiment.log({'test_preds': test_preds})
            self.trainer.logger.experiment.log({'test_preds_rounded': test_preds_rounded})
            self.trainer.logger.experiment.log({'test_labels': test_labels})
        else:  # Assume we are instead using the TensorBoardLogger
            for i, (t_preds, t_preds_rounded, t_labels) in enumerate(zip(test_preds, test_preds_rounded, test_labels)):
                self.logger.experiment.add_image('test_preds', t_preds, i, dataformats='HW')
                self.logger.experiment.add_image('test_preds_rounded', t_preds_rounded, i, dataformats='HW')
                self.logger.experiment.add_image('test_labels', t_labels, i, dataformats='HW')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Lightning calls this inside the predict loop."""
        # Make predictions for a batch of protein complexes
        graph1, graph2 = batch[0], batch[1]
        # Forward propagate with network layers - (batch_size x self.num_channels x len(graph1) x len(graph2))
        logits_list, g1_nf, g1_ef, g2_nf, g2_ef = self.shared_step(graph1, graph2, return_representations=True)
        return logits_list, g1_nf, g1_ef, g2_nf, g2_ef

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-8, verbose=True),
                "monitor": self.metric_to_track,
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # -----------------
        # Model arguments
        # -----------------
        parser.add_argument('--num_gnn_hidden_channels', type=int, default=512,
                            help='Dimensionality of GNN filters (for nodes and edges alike after embedding)')
        parser.add_argument('--num_gnn_attention_heads', type=int, default=32,
                            help='How many multi-head GNN attention blocks to run in parallel')
        parser.add_argument('--interact_module_type', type=str, default='dil_resnet',
                            help='Which type of dense prediction interaction module to use'
                                 ' (i.e. dil_resnet for ResNet2DInputWithOptAttention, or deeplab for DeepLabV3Plus)')
        parser.add_argument('--num_interact_hidden_channels', type=int, default=128,
                            help='Dimensionality of interaction module filters')
        parser.add_argument('--use_interact_attention', action='store_true', dest='use_interact_attention',
                            help='Whether to employ attention in, for example, a Dilated ResNet')
        parser.add_argument('--num_interact_attention_heads', type=int, default=4,
                            help='How many multi-head interact attention blocks to use in parallel')
        parser.add_argument('--weight_classes', action='store_true', dest='weight_classes',
                            help='Whether to use class weighting in our training Cross Entropy')
        parser.add_argument('--fine_tune', action='store_true', dest='fine_tune',
                            help='Whether to fine-tune a trained LitGINI on a new dataset')
        parser.add_argument('--left_pdb_filepath', type=str, default='test_data/4heq_l.pdb',
                            help='A filepath to the left input PDB chain')
        parser.add_argument('--right_pdb_filepath', type=str, default='test_data/4heq_r.pdb',
                            help='A filepath to the right input PDB chain')

        return parser