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
# from torch_geometric.nn import GATConv


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

    # def forward(self, g, n_feat, e_feat):
    #     num_nodes = n_feat.size(0)
    #     for i, layer in enumerate(self.layers):
    #         n_feat = layer(g, n_feat)
    #         n_feat = n_feat.reshape(num_nodes, -1)
    #     return n_feat

    def forward(self, g):
        num_nodes = g.number_of_nodes()
        # 节点特征随机，待讨论
        n_feat = torch.randn((num_nodes, self.in_dims[0]))
        # print("..", n_feat)
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
            n_feat = n_feat.reshape(num_nodes, -1)
        return n_feat
    
# 定义边的起点和终点
src = torch.tensor([0, 1, 2, 1, 6])
dst = torch.tensor([1, 2, 0, 6, 1])
# src = torch.tensor([0, 1, 2])
# dst = torch.tensor([1, 2, 0])

# 使用dgl.DGLGraph创建图
g = dgl.graph((src, dst))
# n_feat = torch.randn((7, 64))

core = UnsupervisedGAT(1, 64, 0, 2, 8)
ans = core(g)
# ans = core(g, n_feat, None)
print(ans.size())


class GraphEncoder(nn.Module):
    def __init__(
        self, feature_dim, node_input_dim, node_hidden_dim, output_dim, edge_input_dim=0, num_layers=2, num_heads=8, norm=True
    ):
        super(GraphEncoder, self).__init__()
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

    def forward(self, g, n_feat):
        n_feat = self.linear_in(n_feat)

        e_feat = None
        x = self.gnn(g, n_feat, e_feat)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)

        x = self.projection(x)
        # import pdb
        # pdb.set_trace()
        return x
    
# map_graph = GraphEncoder(1, 64, 64, 64, 0, 2, 8, True)
# output = map_graph(g, n_feat)