# to test the validation of transformer enocoder for ETA prediction
# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import copy
import random
# from logging import getLogger

import sys
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import time
from torch.utils.data import DataLoader
import tqdm

config0={'vocab_size': 60000, 
        'd_model': 768, 'n_layers': 4, 
        'attn_heads': 12, 'mlp_ratio': 4, 
        'dropout': 0.1, 'drop_path': 0.3, 
        'lape_dim': 256, 'attn_drop': 0.1, 
        'type_ln': 'pre', 'future_mask': False, 
        'device': torch.device('cuda:1'), 
        'sample_rate': 0.2, 'add_pe': True, 
        'padding_idx': 0, 'max_position_embedding': 512, 
        'layer_norm_eps': 1e-5, 'padding_index': 0}

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
    
class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_pe = config['add_pe']
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'], padding_idx=config['padding_index'])

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config['max_position_embedding']).expand((1, -1)))

        self.padding_idx = config['padding_idx']
        if self.add_pe:
            self.position_embedding = nn.Embedding(config['max_position_embedding'], config['d_model'], padding_idx=config['padding_idx'])

        self.dropout = nn.Dropout(config['dropout'])
        self.d_model = config['d_model']
        self.LayerNorm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)

        inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

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

    def __init__(self, config):
        """
        Args:
        """
        super().__init__()

        self.config = config

        self.d_model = self.config.get('d_model', 768)
        self.n_layers = self.config.get('n_layers', 4)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.drop_path = self.config.get('drop_path', 0.3)
        self.lape_dim = self.config.get('lape_dim', 256)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        self.type_ln = self.config.get('type_ln', 'pre')
        self.future_mask = self.config.get('future_mask', False)
        self.device = self.config.get('device', torch.device('cuda:1'))
        self.sample_rate = self.config.get('sample_rate', 0.2)
        self.device = self.config.get('device', torch.device('cpu'))
        self.add_pe = self.config.get('add_pe', True)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        # embedding for BERT, sum of ... embeddings
        self.embedding = BERTEmbedding(self.config)

        # multi-layers transformer blocks, deep network
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, 
                              device=self.device) for i in range(self.n_layers)])


    def forward(self, x, padding_masks,
                output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # embedding the indexed sequence to sequence of vectors
        embedding_output = self.embedding(input_ids=x)  # (B, T, d_model)

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


'''
训练部分
'''

# 数据预处理
train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_valid.csv", sep=';', header=0)
test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_test.csv", sep=';', header=0)


# 删去-999的情况
train_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:,1].values]
train_trajectory_data = [x for x in train_trajectory_data if -999 not in x]
train_time_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:, 2].values]
# 有-999的情况出现
train_times_data = []
for x in train_time_data:
    # trap: 计算mape时travel_time过小，会导致mape异常大
    if x[0] == -999:
        # train_times_data.append(0)
        continue
    elif x[-1]== -999:
        # train_times_data.append(x[-2]-x[0])
        continue
    else:
        train_times_data.append(x[-1]-x[0])

# valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [x for x in valid_trajectory_data if -999 not in x]
valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
valid_times_data = []
for x in valid_time_data:
    if x[0] == -999:
        # valid_times_data.append(0)
        continue
    elif x[-1]== -999:
        # valid_times_data.append(x[-2]-x[0])
        continue
    else:
        valid_times_data.append(x[-1]-x[0])

# test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
test_trajectory_data = [x for x in test_trajectory_data if -999 not in x]
test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
test_times_data = []
for x in test_time_data:
    if x[0] == -999:
        # test_times_data.append(0)
        continue
    elif x[-1]== -999:
        # test_times_data.append(x[-2]-x[0])
        continue
    else:
        test_times_data.append(x[-1]-x[0])


class ETADataset:
    def __init__(self, route_data, time_data):
        self.route_list = route_data
        self.time_list = time_data
        self.dataLen = len(self.route_list)

    def __getitem__(self, index):
        route = self.route_list[index]
        travel_time = self.time_list[index]

        return torch.tensor(route), travel_time


    def __len__(self):
        return self.dataLen

    def collate_fn(self, data):
        route = [item[0] for item in data]
        travel_time = [item[1] for item in data]

        route = pad_sequence(route, padding_value=0, batch_first=True)

        return route, torch.tensor(travel_time)
    
class TransformerTimePred(nn.Module):

    def __init__(self, hidden_size=768):
        super(TransformerTimePred, self).__init__()
        self.hidden_size = hidden_size
        self.route_embeddings = BERT(config0)
        self.time_mapping = nn.Linear(hidden_size,1)

    def forward(self, route_input, travel_time):
        route_input = route_input.long().cuda(1)
        padding_mask = route_input.ne(0).cuda(1)
        route_input_embeds = self.route_embeddings(route_input, padding_masks=padding_mask)[0]
        travel_time = travel_time.cuda(1)

        tmp = route_input_embeds.shape

        route_time_pred = self.time_mapping(route_input_embeds).squeeze(-1)
        # print(route_time_pred.sum(1))
        # print(travel_time)

        mape_loss = torch.abs(route_time_pred.sum(1) - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(route_time_pred.sum(1) - travel_time)
        # print(self.route_embeddings.weight.data[0])
        return mape_loss.mean(), mae_loss.mean()

class TransformerTimePred_train():
    def __init__(self):
        train_dataset = ETADataset(route_data = train_trajectory_data, time_data = train_times_data)
        valid_dataset = ETADataset(route_data = valid_trajectory_data, time_data = valid_times_data)
        test_dataset = ETADataset(route_data = test_trajectory_data, time_data = test_times_data)
        self.train_loader = DataLoader(train_dataset, batch_size=64,
                                    collate_fn=train_dataset.collate_fn,
                                    pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=64,
                                    collate_fn=valid_dataset.collate_fn,
                                    pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64,
                                    collate_fn=test_dataset.collate_fn,
                                    pin_memory=True)


        self.eta_model = TransformerTimePred().cuda(1)
        # set learning_rate
        self.optimizer = torch.optim.Adam(self.eta_model.parameters(), lr=5e-5)

        self.min_dict = {}
        self.min_dict['min_valid_mape'] = 1e18
        self.min_dict['min_valid_mae'] = 1e18


    def train(self):
        self.eta_model.train()
        iter = 0
        for input in tqdm.tqdm(self.train_loader):
            mape_loss, mae_loss = self.eta_model(*input)
            self.optimizer.zero_grad()
            mape_loss.backward()
            self.optimizer.step()
            # print(f"Train mape_Loss: {mape_loss.item():.4f}, Train mae_loss: {mae_loss.item():.4f}")
            if ((iter + 1) % 100 == 0):
                valid_mape, valid_mae = self.valid()
                if self.min_dict['min_valid_mape'] > valid_mape:
                    self.min_dict['min_valid_mape'] = valid_mape
                    self.min_dict['min_valid_mae'] = valid_mae
                    if not os.path.exists('/nas/user/wyh/TNC/model/eta_data/'):
                        os.mkdir('/nas/user/wyh/TNC/model/eta_data/')
                    torch.save({
                        'model': self.eta_model.state_dict(),
                        'best_loss': valid_mape,
                        'opt': self.optimizer,
                    }, '/nas/user/wyh/TNC/model/eta_data/model.pth.tar')

                self.eta_model.train()
            if (iter + 1) % 100 == 0:
                print('mape: ', mape_loss.item(), 'mae: ', mae_loss.item(), 'valid mape: ', self.min_dict['min_valid_mape'], ' valid mae: ',self.min_dict['min_valid_mae'])
            iter += 1

    def valid(self):
        with torch.no_grad():
            self.eta_model.eval()
            avg_mape = 0
            avg_mae = 0
            avg_cnt = 0
            for input in tqdm.tqdm(self.valid_loader):
                mape, mae = self.eta_model(*input)
                # trick-1: 去除异常值
                if mape.item() > 2:
                    continue
                else:
                    avg_mape += mape.item()
                    avg_mae += mae.item()
                    avg_cnt += 1
                # avg_mape += mape.item()
                # avg_mae += mae.item()
                # avg_cnt += 1

            print ('valid mape: ', avg_mape / avg_cnt, ' valid mae: ',avg_mae / avg_cnt)
        return avg_mape / avg_cnt, avg_mae / avg_cnt


    def test(self):
        checkpoint = torch.load('/nas/user/wyh/TNC/model/eta_data/model.pth.tar')
        self.eta_model.load_state_dict(checkpoint['model'])

        with torch.no_grad():
            self.eta_model.eval()
            avg_mape = 0
            avg_mae = 0
            avg_cnt = 0
            for input in tqdm.tqdm(self.test_loader):
                mape, mae = self.eta_model(*input)
                # trick-1: 去除异常值
                if mape.item() > 2:
                    continue
                else:
                    avg_mape += mape.item()
                    avg_mae += mae.item()
                    avg_cnt += 1
                # avg_mape += mape.item()
                # avg_mae += mae.item()
                # avg_cnt += 1

                print('test mape: ', avg_mape / avg_cnt, ' test mae: ', avg_mae / avg_cnt)

        return avg_mape / avg_cnt, avg_mae / avg_cnt

if __name__ == '__main__':
    model_train = TransformerTimePred_train()
    num_epochs = 20 # 训练轮数
    print('Training Start')
    for epoch in range(num_epochs):
        print ('epoch: ',epoch)
        model_train.train()

    model_train.test()