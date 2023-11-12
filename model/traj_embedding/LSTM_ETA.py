import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import time
from torch.utils.data import DataLoader

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
        self.route_embeddings = nn.Embedding(60000, hidden_size)
        self.route_embeddings.weight.data[0] = torch.zeros(hidden_size)
        self.time_mapping = nn.Sequential(nn.Linear(hidden_size,1))
        self.model = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True)

    def forward(self, route_input, travel_time):
        route_input = route_input.cuda()
        route_input_embeds = self.route_embeddings(route_input)
        travel_time = travel_time.cuda()
        input_embed = torch.tensor(route_input_embeds).cuda()

        h0 = torch.zeros(1, route_input_embeds.size()[0], self.hidden_size).cuda()
        c0 = torch.zeros(1, route_input_embeds.size()[0], self.hidden_size).cuda()

        outputs,_ = self.model(input_embed, (h0,c0))

        route_time_pred = self.time_mapping(outputs)

        mape_loss = torch.abs(route_time_pred.sum(1) - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(route_time_pred.sum(1) - travel_time)
        return mape_loss.mean(), mae_loss.mean()


if __name__ == '__main__':
    eta_model = RouteLSTMTimePred().cuda()
    train_dataset = ETADataset(route_data = train_trajectory_data, time_data = train_times_data)
    train_loader = DataLoader(train_dataset, batch_size=64,
                                  collate_fn=train_dataset.collate_fn,
                                  pin_memory=True)
    for input in train_loader:
        mape_loss, mae_loss = eta_model(*input)
        print(mape_loss, mae_loss)
        break
