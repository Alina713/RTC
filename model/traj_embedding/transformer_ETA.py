# 2024.01.09 16:00
# to test the validation of transformer enocoder for ETA prediction
# import dgl
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
# import transformers

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
        'd_model': 768, 'n_layers': 1, 
        'attn_heads': 12, 'mlp_ratio': 4, 
        'dropout': 0.1, 'drop_path': 0.3, 
        'lape_dim': 256, 'attn_drop': 0.1, 
        'type_ln': 'pre', 'future_mask': False, 
        'device': torch.device('cuda:0'), 
        'sample_rate': 0.2, 'add_pe': True, 
        'padding_idx': 0, 'max_position_embedding': 1024, 
        'layer_norm_eps': 1e-6, 'padding_index': 0, 
        'add_cls': True, 
        'cls_idx': 57255, 'sep_idx': 57256}
    
class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_pe = config['add_pe']
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'], padding_idx=config['padding_index'])
        self.geo_embedding = LatLongEmbedding(2, config['d_model'])
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config['max_position_embedding']).expand((1, -1)))

        self.padding_idx = config['padding_idx']
        self.cls_idx = config['cls_idx']
        self.sep_idx = config['sep_idx']
        if self.add_pe:
            self.position_embedding = nn.Embedding(config['max_position_embedding'], config['d_model'], padding_idx=config['padding_idx'])

        self.dropout = nn.Dropout(config['dropout'])
        self.d_model = config['d_model']
        self.LayerNorm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, cls_idx = 57255, sep_idx = 57256, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        # 将input_ids去掉头尾元素
        mask_pad = (input_ids.ne(padding_idx)).int()
        mask_cls = (input_ids.ne(cls_idx)).int()
        mask_sep = (input_ids.ne(sep_idx)).int()
        mask = mask_pad & mask_cls & mask_sep
        # trick Shanghai
        cls_indices = torch.where(input_ids.eq(cls_idx), 511, 0)
        sep_indices = torch.where(input_ids.eq(sep_idx), 512, 0)

        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask

        pids = incremental_indices.long() + cls_indices + sep_indices + padding_idx
        
        return pids
    
    def create_latlong_embedding_from_input_ids(self, input_ids, geo_dict):
        '''
        input_ids: [batch_size, seq_len]
        geo_dict: [[lat, long]...[lat, long]]
        '''
        geo_list = []
        for batch in input_ids:
            geo_list_batch = []
            for node_id in batch:
                if (node_id.item() == self.cls_idx 
                    or node_id.item() == self.sep_idx
                    or node_id.item() == self.padding_idx):
                    geo_list_batch.append([0, 0])
                else:
                    geo_list_batch.append(geo_dict[int(node_id)])
            geo_list.append(geo_list_batch)
        geo_info = torch.tensor(geo_list)

        return geo_info

    def forward(
        self,
        input_ids=None,
        geo_dict=None,
        position_ids=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx).to(input_ids.device)


        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        geo_info = self.create_latlong_embedding_from_input_ids(
            input_ids, geo_dict).to(input_ids.device)
        
        geo_embeddings = self.geo_embedding(geo_info)

        embeddings = inputs_embeds + position_embeddings
        embeddings = embeddings + geo_embeddings

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
                                                                      nhead=config['attn_heads'], 
                                                                      dropout=0.1), 
                                                                      num_layers=config['n_layers'])

    def forward(self, route_input, travel_time, geo_dict):
        route_input = route_input.long().cuda()
        geo_dict = geo_dict.cuda()
        route_input_embeds = self.route_embeddings(route_input, geo_dict)
        route_input_embeds = self.model(route_input_embeds)
        travel_time = travel_time.cuda()

        route_time_pred = self.time_mapping(route_input_embeds).squeeze(-1)

        mape_loss = torch.abs(route_time_pred.sum(1) - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(route_time_pred.sum(1) - travel_time)

        return mape_loss.mean(), mae_loss.mean()
    
class LatLongEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatLongEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    
# 读取/nas/user/wyh/dataset/roadnet/Shanghai/nodeOSM.txt文件
# 读取txt文件，返回一个字典，key为node_id，value为经纬度
def read_nodeOSM(path):
    node_dict = {}
    minlat = 1e18
    minlon = 1e18
    maxlat = -1e18
    maxlon = -1e18
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            node_dict[int(line[0])] = [float(line[1]), float(line[2])]
            minlat = min(minlat, float(line[1]))
            minlon = min(minlon, float(line[2]))
            maxlat = max(maxlat, float(line[1]))
            maxlon = max(maxlon, float(line[2]))
    return node_dict, minlat, minlon, maxlat, maxlon

# 将经纬度进行归一化作为embedding
def normalize_latlong(node_dict, minlat, minlon, maxlat, maxlon):
    for key in node_dict.keys():
        node_dict[key][0] = (node_dict[key][0] - minlat) / (maxlat - minlat)
        node_dict[key][1] = (node_dict[key][1] - minlon) / (maxlon - minlon)
    return node_dict


geo_info = read_nodeOSM('/nas/user/wyh/dataset/roadnet/Shanghai/nodeOSM.txt')
geo_dict = geo_info[0]
minlat = geo_info[1]
minlon = geo_info[2]
maxlat = geo_info[3]
maxlon = geo_info[4]
geo_dict = normalize_latlong(geo_dict, minlat, minlon, maxlat, maxlon)
# 保留geo_dict的value部分。并保存为list
geo_dict = list(geo_dict.values())

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
    def __init__(self, route_data, time_data, geo_dict):
        self.route_list = route_data
        self.time_list = time_data
        self.geo_dict = geo_dict
        self.dataLen = len(self.route_list)

    def __getitem__(self, index):
        route = self.route_list[index]
        # add cls & sep 
        # trick for ShangHai
        route = [57255] + route + [57256]
        travel_time = self.time_list[index]
        geo_dict = self.geo_dict

        return torch.tensor(route), travel_time, geo_dict


    def __len__(self):
        return self.dataLen

    def collate_fn(self, data):
        route = [item[0] for item in data]
        travel_time = [item[1] for item in data]
        geo_dict = data[0][2]

        route = pad_sequence(route, padding_value=0, batch_first=True)

        return route, torch.tensor(travel_time), torch.tensor(geo_dict)

class TransformerTimePred_train():
    def __init__(self):
        train_dataset = ETADataset(route_data = train_trajectory_data, time_data = train_times_data, geo_dict = geo_dict)
        valid_dataset = ETADataset(route_data = valid_trajectory_data, time_data = valid_times_data, geo_dict = geo_dict)
        test_dataset = ETADataset(route_data = test_trajectory_data, time_data = test_times_data, geo_dict = geo_dict)
        self.train_loader = DataLoader(train_dataset, batch_size=32,
                                    collate_fn=train_dataset.collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size=32,
                                    collate_fn=valid_dataset.collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=32,
                                    collate_fn=test_dataset.collate_fn)


        self.eta_model = TransformerTimePred(config0).cuda()
        # set learning_rate
        self.optimizer = torch.optim.Adam(self.eta_model.parameters(), lr= 5e-5)

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