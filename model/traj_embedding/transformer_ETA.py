# to test the validation of transformer enocoder for ETA prediction
# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import math
import numpy as np
import copy
import random
# from logging import getLogger

import sys
import os

import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import time
from torch.utils.data import DataLoader
import tqdm

config0={'vocab_size': 60000, 
        'd_model': 768, 'n_layers': 12, 
        'attn_heads': 12, 'mlp_ratio': 4, 
        'dropout': 0.1, 'drop_path': 0.3, 
        'lape_dim': 256, 'attn_drop': 0.1, 
        'type_ln': 'pre', 'future_mask': False, 
        'device': torch.device('cuda:1'), 
        'sample_rate': 0.2, 'add_pe': True, 
        'padding_idx': 0, 'max_position_embedding': 512, 
        'layer_norm_eps': 1e-5, 'padding_index': 0, 
        'add_cls': True}
    
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


class TransformerTimePred(nn.Module):

    def __init__(self, config):
        super(TransformerTimePred, self).__init__()
        self.hidden_size = config['d_model']
        self.route_embeddings = BERTEmbedding(config)
        self.time_mapping = nn.Linear(config['d_model'],1)
        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config['d_model'], 
                                                                      nhead=config['attn_heads']), num_layers=config['n_layers'])

    def forward(self, route_input, travel_time):
        route_input = route_input.long().cuda(1)
        route_input_embeds = self.route_embeddings(route_input)
        travel_time = travel_time.cuda(1)

        route_time_pred = self.time_mapping(route_input_embeds).squeeze(-1)
        # print(route_time_pred.sum(1))
        # print(travel_time)

        mape_loss = torch.abs(route_time_pred.sum(1) - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(route_time_pred.sum(1) - travel_time)
        # print(self.route_embeddings.weight.data[0])
        return mape_loss.mean(), mae_loss.mean()
    


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
    train_times_data.append(x[-1]-x[0])

# valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [x for x in valid_trajectory_data if -999 not in x]
valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
valid_times_data = []
for x in valid_time_data:
    valid_times_data.append(x[-1]-x[0])

# test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
test_trajectory_data = [x for x in test_trajectory_data if -999 not in x]
test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
test_times_data = []
for x in test_time_data:
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

class TransformerTimePred_train():
    def __init__(self):
        train_dataset = ETADataset(route_data = train_trajectory_data, time_data = train_times_data)
        valid_dataset = ETADataset(route_data = valid_trajectory_data, time_data = valid_times_data)
        test_dataset = ETADataset(route_data = test_trajectory_data, time_data = test_times_data)
        self.train_loader = DataLoader(train_dataset, batch_size=32,
                                    collate_fn=train_dataset.collate_fn,
                                    pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=32,
                                    collate_fn=valid_dataset.collate_fn,
                                    pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32,
                                    collate_fn=test_dataset.collate_fn,
                                    pin_memory=True)


        self.eta_model = TransformerTimePred(config0).cuda(1)
        # set learning_rate
        self.optimizer = torch.optim.Adam(self.eta_model.parameters(), lr=2e-4)

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
    num_epochs = 30 # 训练轮数
    print('Training Start')
    for epoch in range(num_epochs):
        print ('epoch: ',epoch)
        model_train.train()

    model_train.test()