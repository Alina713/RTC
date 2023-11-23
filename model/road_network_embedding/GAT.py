#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 15:38
# TODO:

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

import math
import numpy as np
import copy
import random
from logging import getLogger

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """

        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)

        """
        if position_ids is None:
            return self.pe[:, :x.size(1)].detach()
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe
        

class UnsupervisedGAT(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, num_heads
    ):
        super(UnsupervisedGAT, self).__init__()
        assert node_hidden_dim % num_heads == 0
        self.in_dims = [node_input_dim if i == 0 else node_hidden_dim for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [
                GATConv(
                    in_feats=self.in_dims[i],
                    # in_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_feats=node_hidden_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=False,
                    activation=F.leaky_relu if i + 1 < num_layers else None,
                    # 允许路口编号不连续
                    allow_zero_in_degree=True,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat = None):
        num_nodes = n_feat.size(0)
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
            n_feat = n_feat.reshape(num_nodes, -1)

        return n_feat


# 标准化GAT
class GraphEncoder(nn.Module):
    def __init__(
        self, feature_dim, node_input_dim, node_hidden_dim, output_dim, edge_input_dim=0, num_layers=2, num_heads=8, norm=True
    ):
        super(GraphEncoder, self).__init__()
        self.d_model = node_hidden_dim

        self.linear_in = nn.Linear(feature_dim, node_input_dim)

        self.gnn = UnsupervisedGAT(
            node_input_dim=node_input_dim,
            node_hidden_dim=node_hidden_dim,
            edge_input_dim=edge_input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.norm = norm
        self.projection = nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_hidden_dim, output_dim))

    def forward(self, g, n_feat, x):
        '''
        x为轨迹序列 [rn_id1, rn_id2, rn_id3, ...]
        Args:
            g: dgl format of Map
            * ShangHai
            node_features: (vocab_size, fea_dim)
            * ShangHai (57524, 1)
            x: (B, T)
            * ShangHai (64,Sequence length)
        Returns:
            (B, T, d_model)
        '''
        n_feat = self.linear_in(n_feat)

        e_feat = None
        v = self.gnn(g, n_feat, e_feat)
        if self.norm:
            x = F.normalize(v, p=2, dim=-1, eps=1e-5)

        v = self.projection(v) # (vocab_size, d_model)

        batch_size, seq_len = x.shape
        v = v.unsqueeze(0)  # (1, vocab_size, d_model)
        v = v.expand(batch_size, -1, -1)  # (B, vocab_size, d_model)
        v = v.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)

        out_node_fea_emb = v[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb  # (B, T, d_model)


# 定义边的起点和终点
src = torch.tensor([0, 1, 2, 1, 6])
dst = torch.tensor([1, 2, 0, 6, 1])

g = dgl.graph((src, dst))
# 7为图的节点数目，1为节点特征的维度
n_feat = torch.randn((7, 1))
x = torch.tensor([[0, 1, 2, 1, 6], [1, 2, 0, 6, 1]])

map_graph = GraphEncoder(feature_dim=1, node_input_dim=1, node_hidden_dim=64, output_dim=64, edge_input_dim=0, num_layers=2, num_heads=8, norm=False)
output1 = map_graph(g, n_feat, x)
output2 = map_graph(g, n_feat, x)

print(output1)
print(output2)

# torch.Size([7, 64])

class BERTEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """
        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = UnsupervisedGAT(d_model=d_model, in_feature=node_fea_dim,
                                   num_heads_per_layer=gat_heads_per_layer,
                                   num_features_per_layer=gat_features_per_layer,
                                   add_skip_connection=True, bias=True, dropout=gat_dropout,
                                   load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None):
        """
        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
        Returns:
            (B, T, d_model)
        """
        if self.add_gat:
            x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0])  # (B, T, d_model)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        return self.dropout(x)