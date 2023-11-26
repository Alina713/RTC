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

# from torch.autograd import Variable

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
            v = F.normalize(v, p=2, dim=-1, eps=1e-5)

        # 感觉是归一化转接头， 暂时注释
        # v = self.projection(v) # (vocab_size, d_model)

        batch_size, seq_len = x.shape
        v = v.unsqueeze(0)  # (1, vocab_size, d_model)
        v = v.expand(batch_size, -1, -1)  # (B, vocab_size, d_model)
        v = v.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)

        out_node_fea_emb = v[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb  # (B, T, d_model)

class RouteEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, 
                 add_pe=True, node_fea_dim = 1, add_gat=True, norm = True, 
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """
        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = GraphEncoder(feature_dim=node_fea_dim, node_input_dim=16, node_hidden_dim=d_model,
                                                output_dim=d_model, edge_input_dim=0, num_layers=2, num_heads=8, norm=norm)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, g, n_feat, position_ids=None):
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
            x = self.token_embedding(g, n_feat, sequence)  # (B, T, d_model)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        return self.dropout(x)
    
# 定义边的起点和终点
src = torch.tensor([0, 1, 2, 1, 6])
dst = torch.tensor([1, 2, 0, 6, 1])

g = dgl.graph((src, dst))
# 7为图的节点数目，1为节点特征的维度
n_feat = torch.randn((7, 1))
x = torch.tensor([[0, 1, 2, 1, 6], [1, 2, 0, 6, 1]])

# GraphEncoder里面有问题
a = RouteEmbedding(64, dropout=0.1, add_pe=True, node_fea_dim=1, add_gat=True, norm=True)
ans = a(x, g, n_feat)

print(ans.shape)
print(ans)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)
        Returns:
        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """Position-wise Feed-Forward Networks
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        """
        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)
        Returns:
            (B, T, d_model)
        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score