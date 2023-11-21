import sys
import os
sys.path.append("/nas/user/wyh/TNC/")
from pybind.test_funcs import Map
from model.road_network_embedding.GAT import GraphEncoder

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import time
from torch.utils.data import DataLoader
import tqdm

SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
dgl_SH_map = SH_map.dgl_valid_map()
n_feat = torch.randn((57254, 1))

# 1 或许可以换成2 表示为经纬度特征
map_graph = GraphEncoder(1, 64, 64, 64, 0, 2, 8, False)
# output = map_graph(dgl_SH_map, n_feat)

# print(output.size())

# 数据预处理
train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_valid.csv", sep=';', header=0)
test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_test.csv", sep=';', header=0)

# 将字符串转换为数字列表，并计算每个序列的长度
# train_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:,1].values]
train_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:,1].values]
train_time_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:, 2].values]
# 有-999的情况出现
train_times_data = []
for x in train_time_data:
    # trap: 计算mape时travel_time过小，会导致mape异常大
    if x[0] == -999:
        train_times_data.append(0)
    elif x[-1]== -999:
        train_times_data.append(x[-2]-x[0])
    else:
        train_times_data.append(x[-1]-x[0])

# valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
valid_times_data = []
for x in valid_time_data:
    if x[0] == -999:
        valid_times_data.append(0)
    elif x[-1]== -999:
        valid_times_data.append(x[-2]-x[0])
    else:
        valid_times_data.append(x[-1]-x[0])

# test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
test_times_data = []
for x in test_time_data:
    if x[0] == -999:
        test_times_data.append(0)
    elif x[-1]== -999:
        test_times_data.append(x[-2]-x[0])
    else:
        test_times_data.append(x[-1]-x[0])


# 路口经纬度文件：/nas/user/wyh/dataset/roadnet/Shanghai/nodeOSM.txt
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


class RouteLSTMTimePred(nn.Module):

    def __init__(self, hidden_size=64):
        super(RouteLSTMTimePred, self).__init__()
        self.hidden_size = hidden_size
        # self.route_embeddings = nn.Embedding(60000, hidden_size)
        self.route_embeddings = map_graph(dgl_SH_map, n_feat).cuda()
        # self.route_embeddings.weight.data[0] = torch.zeros(hidden_size)
        self.time_mapping = nn.Linear(hidden_size,1)
        self.model = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True)

    def forward(self, route_input, travel_time):
        route_input = route_input.long().cuda()
        route_input_embeds = self.route_embeddings[route_input]
        travel_time = travel_time.cuda()

        h0 = torch.zeros(1, route_input_embeds.size()[0], self.hidden_size).cuda()
        c0 = torch.zeros(1, route_input_embeds.size()[0], self.hidden_size).cuda()

        outputs,_ = self.model(route_input_embeds, (h0,c0))

        route_time_pred = self.time_mapping(outputs).squeeze(-1)
        # print(route_time_pred.sum(1))
        # print(travel_time)

        mape_loss = torch.abs(route_time_pred.sum(1) - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(route_time_pred.sum(1) - travel_time)
        # print(self.route_embeddings.weight.data[0])
        return mape_loss.mean(), mae_loss.mean()

class RouteLSTMTimePred_train():
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


        self.eta_model = RouteLSTMTimePred().cuda()
        # ignored_params = list(map(id, self.eta_model.route_embeddings.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, self.eta_model.parameters())
        # params_list = [{'params': base_params, 'lr': 0.01}]
        # set learning_rate
        # self.optimizer = torch.optim.Adam(params_list, lr=0.01)
        self.optimizer = torch.optim.Adam(self.eta_model.parameters(), lr=0.01)

        # self.eta_model = RouteLSTMTimePred(bert_model=model).cuda()

        # self.optim = torch.optim.Adam(params_list, lr=args.learning_rate)

        self.min_dict = {}
        self.min_dict['min_valid_mape'] = 1e18
        self.min_dict['min_valid_mae'] = 1e18


    def train(self):
        self.eta_model.train()
        iter = 0
        for input in tqdm.tqdm(self.train_loader):
            mape_loss, mae_loss = self.eta_model(*input)
            self.optimizer.zero_grad()
            mape_loss.backward(retain_graph=True)
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
                    }, '/nas/user/wyh/TNC/model/eta_data/gat.model.pth.tar')

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
        checkpoint = torch.load('/nas/user/wyh/TNC/model/eta_data/gat.model.pth.tar')
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
    model_train = RouteLSTMTimePred_train()
    num_epochs = 30 # 训练轮数
    print('Training Start')
    for epoch in range(num_epochs):
        print ('epoch: ',epoch)
        model_train.train()

    model_train.test()