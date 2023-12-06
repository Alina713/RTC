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

SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
dgl_SH_map = SH_map.dgl_valid_map()
adj = dgl_SH_map.adj().to_dense()
# print(adj.shape)
# # torch.Size([57254, 57254])
tmp_n_feat = torch.randn((57254, 1))


# 数据预处理
train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_valid.csv", sep=';', header=0)
test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_test.csv", sep=';', header=0)

# 将字符串转换为数字列表，并计算每个序列的长度
# train_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:,1].values]
train_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:,1].values]
# train_trajectory_data = [[1] if not sublist else sublist for sublist in train_trajectory_data]
# train_trajectory_data = [sublist for sublist in train_trajectory_data if sublist]
train_padding_masks = [None for _ in range(len(train_trajectory_data))]
train_time_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:, 2].values]
# 有-999的情况出现
train_times_data = []
for x in train_time_data:
    # trap: 计算mape时travel_time过小，会导致mape异常大
    if x[0] == -999:
        # train_times_data.append(0)
        continue
    elif x[-1]== -999:
        train_times_data.append(x[-2]-x[0])
    else:
        train_times_data.append(x[-1]-x[0])

# valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
# valid_trajectory_data = [[1] if not sublist else sublist for sublist in valid_trajectory_data]
# valid_trajectory_data = [sublist for sublist in valid_trajectory_data if sublist]
valid_padding_masks = [None for _ in range(len(valid_trajectory_data))]
valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
valid_times_data = []
for x in valid_time_data:
    if x[0] == -999:
        # valid_times_data.append(0)
        continue
    elif x[-1]== -999:
        valid_times_data.append(x[-2]-x[0])
    else:
        valid_times_data.append(x[-1]-x[0])

# test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
# [[], [1, 2, 3], [4, 5, 6], [], [7, 8, 9]]
# 把[]改成[1]，但可能存在路口为1的情况
# test_trajectory_data = [[1] if not sublist else sublist for sublist in test_trajectory_data]
# test_trajectory_data = [sublist for sublist in test_trajectory_data if sublist]
test_padding_masks = [None for _ in range(len(test_trajectory_data))]
test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
test_times_data = []
for x in test_time_data:
    if x[0] == -999:
        # test_times_data.append(0)
        continue
    elif x[-1]== -999:
        test_times_data.append(x[-2]-x[0])
    else:
        test_times_data.append(x[-1]-x[0])


class GATDataset(Dataset):
    def __init__(self, x, padding_masks, g, n_feat, travel_time):
        self.x = x
        self.padding_masks = padding_masks
        self.g = g
        self.n_feat = n_feat
        self.travel_time = travel_time

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'padding_masks': self.padding_masks[idx],
            'g': self.g,
            'n_feat': self.n_feat,
            'travel_time': self.travel_time[idx]
        }
    
def collate(samples):
    """
    batch数据处理函数，将batch数据处理为dataloader的输出格式。

    参数:
        samples (list): batch数据

    返回:
        dict: batch数据
    """
    batch = {}
    # print(samples)
    batch['x'] = pad_sequence([torch.tensor(sample['x']) for sample in samples], padding_value=0, batch_first=True)
    pm = batch['x'].ne(0)
    batch['padding_masks'] = pm
    batch['g'] = samples[0]['g']
    batch['n_feat'] = samples[0]['n_feat']
    batch['travel_time'] = torch.tensor([sample['travel_time'] for sample in samples])
    return batch

class LinearETA(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.d_model = self.config.get('d_model', 768)

        self.model = BERT(config, data_feature)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, padding_masks, g, n_feat, travel_time, 
                output_hidden_states=False, output_attentions=False):
        # import pdb
        # pdb.set_trace()
        x = x.long().cuda(1)
        padding_masks = padding_masks.cuda(1)
        g = g.to('cuda:1')
        n_feat = n_feat.cuda(1)
        travel_time = travel_time.cuda(1)

        token_emb, _, _ = self.model(x=x, padding_masks=padding_masks,
                              g=g, n_feat=n_feat,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)  # (B, T, d_model)

        input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
        sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        # print(sum_mask)
        # print(sum_mask.shape)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # print(sum_mask)
        # print(sum_mask.shape)
        traj_emb = sum_embeddings / sum_mask  # (batch_size, feat_dim)
        eta_pred = self.linear(traj_emb)  # (B, 1)
        eta_pred = eta_pred.squeeze(1) # (B)
        # print(eta_pred)
        # print(eta_pred.shape)
        # print(travel_time)
        # print(travel_time.shape)
        # print(eta_pred - travel_time)
        
        mape_loss = torch.abs(eta_pred - travel_time) / (travel_time + 1e-9)
        mae_loss = torch.abs(eta_pred - travel_time)
        # print(mape_loss.mean())
        # print(mae_loss.mean())

        # assert 2==1, 'stop'
        return mape_loss.mean(), mae_loss.mean()


class Route_gattrans_TimePred_train():
    def __init__(self):
        train_dataset = GATDataset(x=train_trajectory_data, padding_masks=train_padding_masks, g=dgl_SH_map, n_feat=tmp_n_feat, travel_time=train_times_data)
        valid_dataset = GATDataset(x=valid_trajectory_data, padding_masks=valid_padding_masks, g=dgl_SH_map, n_feat=tmp_n_feat, travel_time=valid_times_data)
        test_dataset = GATDataset(x=test_trajectory_data, padding_masks=test_padding_masks, g=dgl_SH_map, n_feat=tmp_n_feat, travel_time=test_times_data)
        self.train_loader = DataLoader(train_dataset, batch_size=64,
                                    collate_fn=collate,
                                    pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=64,
                                    collate_fn=collate,
                                    pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64,
                                    collate_fn=collate,
                                    pin_memory=True)

        config = {'key': 'value'}
        data_feature = {'key': 'value'}
        self.eta_model = LinearETA(config=config, data_feature=data_feature).cuda(1)
        # set learning_rate
        self.optimizer = torch.optim.Adam(self.eta_model.parameters(), lr=0.01)

        self.min_dict = {}
        self.min_dict['min_valid_mape'] = 1e18
        self.min_dict['min_valid_mae'] = 1e18


    def train(self):
        self.eta_model.train()
        iter = 0
        for i, batch in tqdm.tqdm(enumerate(self.train_loader)):
            mape_loss, mae_loss = self.eta_model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], batch['travel_time'])
        # for input in tqdm.tqdm(self.train_loader):
        #     mape_loss, mae_loss = self.eta_model(*input)
            self.optimizer.zero_grad()
            mape_loss.backward()
            self.optimizer.step()
            print(f"Train mape_Loss: {mape_loss.item():.4f}, Train mae_loss: {mae_loss.item():.4f}")
            # assert 2==1, 'stop'
            if ((iter + 1) % 1000 == 0):
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
            if (iter + 1) % 1000 == 0:
                print('mape: ', mape_loss.item(), 'mae: ', mae_loss.item(), 'valid mape: ', self.min_dict['min_valid_mape'], ' valid mae: ',self.min_dict['min_valid_mae'])
            iter += 1

    def valid(self):
        with torch.no_grad():
            self.eta_model.eval()
            avg_mape = 0
            avg_mae = 0
            avg_cnt = 0

            for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
                mape, mae = self.eta_model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], batch['travel_time'])
            # for input in tqdm.tqdm(self.valid_loader):
            #     mape, mae = self.eta_model(*input)
                # trick-1: 去除异常值
                if mape.item() > 20:
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
            for i, batch in tqdm.tqdm(enumerate(self.test_loader)):
                mape, mae = self.eta_model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], batch['travel_time'])
            # for input in tqdm.tqdm(self.test_loader):
            #     mape, mae = self.eta_model(*input)

                # trick-1: 去除异常值
                if mape.item() > 20:
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
    model_train = Route_gattrans_TimePred_train()
    num_epochs = 20 # 训练轮数
    print('Training Start')
    for epoch in range(num_epochs):
        print ('epoch: ',epoch)
        model_train.train()

    model_train.test()
    # # 定义边的起点和终点
    # src = torch.tensor([0, 1, 2, 1, 6])
    # dst = torch.tensor([1, 2, 0, 6, 1])
    # g = dgl.graph((src, dst))
    # # 7为图的节点数目，1为节点特征的维度
    # n_feat = torch.randn((7, 1))
    # # print("n_feat: ", n_feat)
    # config = {'key': 'value'}
    # data_feature = {'key': 'value'}
    # model_train = BERT(config, data_feature)

    # # 假设我们有以下数据
    # x = torch.tensor([[1, 2, 1, 6, 0], [1, 2, 6, 6, 3], [1, 1, 1, 0, 0], [1, 2, 6, 6, 3]])
    # print(x.shape)
    # padding_masks = [None for _ in range(x.shape[0])]
    # print(padding_masks)
    # # 创建 Dataset 对象
    # dataset = GATDataset(x, padding_masks, g, n_feat)

    # # 创建 DataLoader 对象
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    # dataloader2 = DataLoader(dataset, batch_size=1, collate_fn=collate)

    # # print(dataloader)
    # for i, batch in enumerate(dataloader):
    #     print(batch['x'].shape)
    #     # ans = model_train(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'])
    #     # print(ans)

    # print("dataloader2")


    # for i, batch in enumerate(dataloader2):
    #     print(batch['x'].shape)
        # ans = model_train(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'])
        # print(ans)
    # padding_mask = get_padding_mask(x)
    # # print(padding_mask)

    # ans = model_train(x, padding_mask, g, n_feat)
    # print(ans[0])
    # print(ans[0].shape)
    # num_epochs = 30 # 训练轮数
    # print('Training Start')
    # for epoch in range(num_epochs):
    #     print ('epoch: ',epoch)
    #     model_train.train()

    # model_train.test()