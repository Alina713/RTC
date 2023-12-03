import sys
import os
sys.path.append("/nas/user/wyh/TNC/")
sys.path.append("/nas/user/wyh/TNC/model/road_network_embedding/")
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

import dgl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from GAT import BERT
from GAT import RouteEmbedding

# SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
# dgl_SH_map = SH_map.dgl_valid_map()
# adj = dgl_SH_map.adj().to_dense()
# # print(adj.shape)
# # # torch.Size([57254, 57254])
# tmp_n_feat = torch.randn((57254, 1))


# # 1 或许可以换成2 表示为经纬度特征
# # map_graph = GraphEncoder(1, 64, 64, 64, 0, 2, 8, False)
# # output = map_graph(dgl_SH_map, n_feat)

# # print(output.size())

# # 数据预处理
# train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_train.csv", sep=';', header=0)
# valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_valid.csv", sep=';', header=0)
# test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_test.csv", sep=';', header=0)

# # 将字符串转换为数字列表，并计算每个序列的长度
# # train_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:,1].values]
# train_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:,1].values]
# train_time_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:, 2].values]
# # 有-999的情况出现
# train_times_data = []
# for x in train_time_data:
#     # trap: 计算mape时travel_time过小，会导致mape异常大
#     if x[0] == -999:
#         train_times_data.append(0)
#     elif x[-1]== -999:
#         train_times_data.append(x[-2]-x[0])
#     else:
#         train_times_data.append(x[-1]-x[0])

# # valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
# valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
# valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
# valid_times_data = []
# for x in valid_time_data:
#     if x[0] == -999:
#         valid_times_data.append(0)
#     elif x[-1]== -999:
#         valid_times_data.append(x[-2]-x[0])
#     else:
#         valid_times_data.append(x[-1]-x[0])

# # test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
# test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
# test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
# test_times_data = []
# for x in test_time_data:
#     if x[0] == -999:
#         test_times_data.append(0)
#     elif x[-1]== -999:
#         test_times_data.append(x[-2]-x[0])
#     else:
#         test_times_data.append(x[-1]-x[0])


# class GATDataset(Dataset):
#     def __init__(self, x, padding_masks, g, n_feat):
#         self.x = x
#         self.padding_masks = padding_masks
#         self.g = g
#         self.n_feat = n_feat

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return {
#             'x': self.x[idx],
#             'padding_masks': self.padding_masks[idx],
#             'g': self.g[idx],
#             'n_feat': self.n_feat[idx]
#         }


# def create_data_loader(dataset, batch_size):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )

# dataset = GATDataset(x, padding_masks, g, n_feat)
# data_loader = create_data_loader(dataset, batch_size=16)

def get_padding_mask(time_series, pad_value=0):
    """
    根据时间序列获取padding mask。

    参数:
        time_series (np.array): 时间序列，假设填充值为0
        pad_value (int): 填充值，默认为0

    返回:
        np.array: padding mask，和时间序列形状相同，填充位置为True，非填充位置为False
    """
    return time_series != pad_value


if __name__ == '__main__':
    # 定义边的起点和终点
    src = torch.tensor([0, 1, 2, 1, 6])
    dst = torch.tensor([1, 2, 0, 6, 1])
    g = dgl.graph((src, dst))
    # 7为图的节点数目，1为节点特征的维度
    n_feat = torch.randn((7, 1))
    x = torch.tensor([[1, 2, 1, 6, 2], [1, 2, 6, 6, 1]])

    # map_graph = GraphEncoder(feature_dim=1, node_input_dim=1, node_hidden_dim=64, output_dim=64, edge_input_dim=0, num_layers=2, num_heads=8, norm=False)
    # map_graph = RouteEmbedding(d_model=64,dropout=0.1, add_pe=True,node_fea_dim=1, add_gat=True, norm = True)
    # output1 = map_graph(x, g, n_feat)
    # output2 = map_graph(x, g, n_feat)

    # print(output1)
    # print(output2)
    config = {'key': 'value'}
    data_feature = {'key': 'value'}
    model_train = BERT(config, data_feature)

    padding_mask = get_padding_mask(x)
    # print(padding_mask)

    ans = model_train(x, padding_mask, g, n_feat)
    print(ans[0])
    print(ans[0].shape)
    # num_epochs = 30 # 训练轮数
    # print('Training Start')
    # for epoch in range(num_epochs):
    #     print ('epoch: ',epoch)
    #     model_train.train()

    # model_train.test()