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
# from dgl.nn.pytorch import GATv2Conv
# from dgl.nn.pytorch import TransformerConv

import math
import numpy as np
import copy
import random
from logging import getLogger


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)

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
        # self.in_dims = [node_input_dim if i == 0 else node_hidden_dim for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [
                GATConv(
                    # in_feats=self.in_dims[i],
                    in_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_feats=node_hidden_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.6,
                    residual=True,
                    activation=F.leaky_relu if i + 1 < num_layers else None,
                    # activation = nn.ELU() if i + 1 < num_layers else None,
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
        self, feature_dim, node_input_dim, node_hidden_dim, output_dim, 
        edge_input_dim=0, num_layers=2, num_heads=8, norm=True
    ):
        super(GraphEncoder, self).__init__()
        self.d_model = node_hidden_dim

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
        
        # g的number_of_nodes()是路口的数量 57254
        self.fea_encoder = nn.Embedding(57254, node_input_dim)

    def forward(self, g, n_feat, x):
        '''
        x为轨迹序列 [rn_id1, rn_id2, rn_id3, ...]
        Args:
            g: dgl format of Map
            * ShangHai
            node_features: (vocab_size, fea_dim)
            * ShangHai (57254, 1)
            x: (B, T)
            * ShangHai (64,Sequence length)
        Returns:
            (B, T, d_model)
        '''
        n_feat = self.fea_encoder(n_feat)

        e_feat = None
        v = self.gnn(g, n_feat, e_feat)
        ab = v.shape
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
                 add_pe=True, node_fea_dim = 1, add_gat=True, norm = True):
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
            self.token_embedding = GraphEncoder(feature_dim=node_fea_dim, node_input_dim=64, node_hidden_dim=d_model,
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

# 以上可以得到一个batch中的positional embedding + token embedding
# size: (B, T, d_model)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), 
                # device = torch.device('cuda:0')
                 ):
        super().__init__()

        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False):
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
                 device=torch.device('cpu')):
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
                                              device=device)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False):
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
                                                  future_mask=future_mask, output_attentions=output_attentions)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score
    

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config, data_feature):
        """
        Args:
        """
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size', 2)
        # self.node_fea_dim = data_feature.get('node_fea_dim')
        self.node_fea_dim = data_feature.get('node_fea_dim', 1)

        # d_model 必须可以整除 attn_heads
        self.d_model = self.config.get('d_model', 768)
        self.n_layers = self.config.get('n_layers', 12)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.drop_path = self.config.get('drop_path', 0.3)
        self.lape_dim = self.config.get('lape_dim', 256)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        # pre改为post
        self.type_ln = self.config.get('type_ln', 'post')
        # 如果 `add_cls` 为 `True`，则在生成 `future_mask` 时，可能会将 `[CLS]` token 对应的位置设置为 `False`，即允许模型在生成 `[CLS]` token 的输出时，
        # 查看整个输入序列的信息。这是因为 `[CLS]` token 的输出通常被用作整个序列的聚合表示，所以需要考虑整个序列的信息。
        # False， 改
        self.future_mask = self.config.get('future_mask', False)
        self.add_cls = self.config.get('add_cls', True)
        self.device = self.config.get('device', torch.device('cuda:1'))
        self.add_pe = self.config.get('add_pe', True)
        self.add_gat = self.config.get('add_gat', True)
        self.norm = self.config.get('norm', True)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        # embedding for BERT, sum of ... embeddings
        self.embedding = RouteEmbedding(d_model=self.d_model, dropout=self.dropout,
                                       add_pe=self.add_pe, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat, norm=self.norm)

        # multi-layers transformer blocks, deep network
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device) for i in range(self.n_layers)])


    def forward(self, x, padding_masks, g, n_feat, 
                output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            `padding_masks`是一个形状为`(batch_size, seq_length)`的张量，其中`1`表示该位置的向量是需要保留的，`0`表示该位置是填充的。
            g 与 n_feat对应原来的 graph_dict
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        position_ids = None


        # embedding the indexed sequence to sequence of vectors
        embedding_output = self.embedding(sequence=x, g = g, n_feat = n_feat, position_ids = None)  # (B, T, d_model)


        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
        all_hidden_states = [embedding_output] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append(embedding_output)
            if output_attentions:
                all_self_attentions.append(attn_score)
        return embedding_output, all_hidden_states, all_self_attentions  # (B, T, d_model), list of (B, T, d_model), list of (B, head, T, T)


class MLPLayer(nn.Module):
    """
    服务于cls
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    GitHub Copilot: 这段代码定义了一个名为`MLPLayer`的PyTorch模块，它是一个简单的多层感知机（MLP）层。
    以下是每一行代码的解释：
    1. `class MLPLayer(nn.Module):` 定义了一个新的类，名为`MLPLayer`，它继承自`nn.Module`。`nn.Module`是PyTorch中所有神经网络模块的基类。
    2. `def __init__(self, d_model):` 定义了类的初始化函数，它接受一个参数`d_model`，表示输入和输出的特征维度。
    3. `super().__init__()` 调用父类的初始化函数，这是在定义新的PyTorch模块时的标准做法。
    4. `self.dense = nn.Linear(d_model, d_model)` 定义了一个线性层，它的输入和输出特征维度都是`d_model`。
    5. `self.activation = nn.Tanh()` 定义了一个激活函数，这里使用的是双曲正切函数。
    6. `def forward(self, features):` 定义了前向传播函数，它接受一个参数`features`，表示输入的特征。
    7. `x = self.dense(features)` 将输入的特征通过线性层进行变换。
    8. `x = self.activation(x)` 将线性层的输出通过激活函数进行变换。
    9. `return x` 返回变换后的特征。
    总的来说，这个`MLPLayer`模块就是一个包含一个线性层和一个激活函数的简单神经网络层，它可以将输入的特征进行一次线性变换和非线性变换。
    """

    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BERTPooler(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.pooling = self.config.get('pooling', 'mean')
        self.add_cls = self.config.get('add_cls', True)
        self.d_model = self.config.get('d_model', 768)
        self.linear = MLPLayer(d_model=self.d_model)

        self._logger = getLogger()
        self._logger.info("Building BERTPooler model")

    def forward(self, bert_output, padding_masks, hidden_states=None):
        """
        Args:
            bert_output: (batch_size, seq_length, d_model) torch tensor of bert output
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            hidden_states: list of hidden, (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, feat_dim) feat_dim一般为d_model768

        1. **max**：这种策略是对每个位置的向量取最大值。首先，将填充位置的向量设置为负无穷大，然后对每个位置的向量取最大值。这种策略的假设是，
        最大值能够捕获到最重要的信息。
        2. **avg_first_last**：这种策略是取第一个和最后一个位置的向量的平均值。首先，计算第一个和最后一个位置的向量的平均值，然后将这个平均值与
        一个掩码相乘，将填充位置的值设置为0，然后对每个位置的值求和，最后将和除以非填充位置的数量。这种策略的假设是，第一个和最后一个位置的向量能够捕获到整个序列的信息。
        3. **avg_top2**：这种策略是取最后两个位置的向量的平均值。具体的操作与`avg_first_last`相同，只是取的是最后两个位置的向量的平均值。这种策略的假设是，最后两
        个位置的向量能够捕获到整个序列的信息。
        这三种策略的主要区别在于它们选择的位置和操作方式。具体选择哪种策略，可能需要根据你的任务和数据进行实验来确定。
        """
        token_emb = bert_output  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls':
            if self.add_cls:
                return self.linear(token_emb[:, 0, :])  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "all":
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            return sum_embeddings  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))



def get_padding_mask(time_series, pad_value=0):
    """
    根据时间序列获取padding mask。

    参数:
        time_series (np.array): 时间序列，假设填充值为0
        pad_value (int): 填充值，默认为0

    返回:
        np.array: padding mask，和时间序列形状相同，填充位置为False，非填充位置为True
    """
    return time_series != pad_value


if __name__ == '__main__':
    src = torch.tensor([3, 1, 2, 1, 6])
    dst = torch.tensor([1, 2, 3, 6, 1])
    g = dgl.graph((src, dst))
    n_feat = torch.randn((7, 1))
    x = torch.tensor([[2, 6, 3, 1, 0], [1, 2, 1, 0, 0]])
    model = UnsupervisedGAT(node_input_dim=1, node_hidden_dim=16, edge_input_dim=0, num_layers=2, num_heads=8)
    ans = model(g, n_feat)
    print(ans.shape)
    print(ans)
    # config = {'key': 'value'}
    # data_feature = {'key': 'value'}
    # model_train = BERT(config, data_feature)

    # padding_mask = get_padding_mask(x)

    # ans = model_train(x, padding_mask, g, n_feat)
    # print(ans)
    # print(ans[0].shape)