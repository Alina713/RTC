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
from GAT import BERTPooler

import torch.nn.functional as F

# Dataloader书写
def compute_degree_features(graph):
    """
    Args:
        graph: dgl graph
    Returns:
        in_degree: (vocab_size, )
        out_degree: (vocab_size, )
    """
    in_degree = graph.in_degrees(range(graph.number_of_nodes())).float().unsqueeze(1)  # (vocab_size, 1)
    out_degree = graph.out_degrees(range(graph.number_of_nodes())).float().unsqueeze(1)  # (vocab_size, 1)
    degree_features = torch.cat([in_degree, out_degree], dim=1)  # (vocab_size, 2)
    return degree_features

# 原始样本
SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
dgl_SH_map = SH_map.dgl_valid_map()

tmp_n_feat = compute_degree_features(dgl_SH_map)

# 数据预处理
train_data = pd.read_csv("/nas/user/wyh/TNC/data/align_data/SH/SH_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/align_data/SH/SH_valid.csv", sep=';', header=0)
# test_data = pd.read_csv("/nas/user/wyh/TNC/data/tiny_ETA/SHmap_test.csv", sep=';', header=0)


train_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in train_data.iloc[:,1].values]
# train_padding_masks = [None for _ in range(len(train_trajectory_data))]


valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in valid_data.iloc[:,1].values]
# valid_padding_masks = [None for _ in range(len(valid_trajectory_data))]


# test_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in test_data.iloc[:,1].values]
# [[], [1, 2, 3], [4, 5, 6], [], [7, 8, 9]]
# test_padding_masks = [None for _ in range(len(test_trajectory_data))]


# 正样本aug
aug_SH_map = Map("/nas/user/wyh/TNC/traj_dealer/diff_map", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
aug_dgl_SH_map = aug_SH_map.dgl_valid_map()

aug_tmp_n_feat = compute_degree_features(aug_dgl_SH_map)

aug_train_data = pd.read_csv("/nas/user/wyh/TNC/data/align_data/SH/SH_aug_train.csv", sep=';', header=0)
aug_valid_data = pd.read_csv("/nas/user/wyh/TNC/data/align_data/SH/SH_aug_valid.csv", sep=';', header=0)
# aug_test_data = pd.read_csv("/nas/user/wyh/TNC/data/tiny_ETA/diff1_map/SHmap_test.csv", sep=';', header=0)

aug_train_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in aug_train_data.iloc[:,1].values]
# aug_train_padding_masks = [None for _ in range(len(aug_train_trajectory_data))]

aug_valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in aug_valid_data.iloc[:,1].values]
# aug_valid_padding_masks = [None for _ in range(len(aug_valid_trajectory_data))]

# aug_test_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in aug_test_data.iloc[:,1].values]
# aug_test_padding_masks = [None for _ in range(len(aug_test_trajectory_data))]


class GATDataset(Dataset):
    def __init__(self, x, g, n_feat, aug_x, aug_g, aug_n_feat):
        self.x = x
        self.g = g
        self.n_feat = n_feat
        self.aug_x = aug_x
        self.aug_g = aug_g
        self.aug_n_feat = aug_n_feat

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': [2]+self.x[idx],
            'g': self.g,
            'n_feat': self.n_feat,
            'aug_x': [2] + self.aug_x[idx],
            'aug_g': self.aug_g,
            'aug_n_feat': self.aug_n_feat, 
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
    batch['aug_x'] = pad_sequence([torch.tensor(sample['aug_x']) for sample in samples], padding_value=0, batch_first=True)
    aug_pm = batch['aug_x'].ne(0)
    batch['aug_padding_masks'] = aug_pm
    batch['aug_g'] = samples[0]['aug_g']
    batch['aug_n_feat'] = samples[0]['aug_n_feat']

    return batch

# train_dataset = GATDataset(x=train_trajectory_data, g=dgl_SH_map, n_feat=tmp_n_feat, 
#                            aug_x=aug_train_trajectory_data, aug_g=aug_dgl_SH_map, 
#                            aug_n_feat=aug_tmp_n_feat)
# valid_dataset = GATDataset(x=valid_trajectory_data, g=dgl_SH_map, n_feat=tmp_n_feat,
#                            aug_x=aug_train_trajectory_data, aug_g=aug_dgl_SH_map, 
#                            aug_n_feat=aug_tmp_n_feat)
# # print(train_dataset[0])


# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
# # print(train_dataloader)
# valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True, collate_fn=collate)

# 执行器书写，计算loss
class BERTContrastive(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.pooling = self.config.get('pooling', 'cls')

        self.bert = BERT(config, data_feature)
        self.pooler = BERTPooler(config, data_feature)

    def forward(self, x, padding_masks, g, n_feat, aug_x, aug_padding_masks, aug_g, aug_n_feat):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, d_model)
        """
        x = x.cuda()
        padding_masks = padding_masks.cuda()
        g = g.to('cuda:0')
        n_feat = n_feat.cuda()
        aug_x = aug_x.cuda()
        aug_padding_masks = aug_padding_masks.cuda()
        aug_g = aug_g.to('cuda:0')
        aug_n_feat = aug_n_feat.cuda()

        x1, _, _ = self.bert(x=x, padding_masks=padding_masks,g=g, n_feat=n_feat)  # (B, T, d_model)
        x1 = self.pooler(bert_output=x1, padding_masks=padding_masks)  # (B, d_model)

        x2, _, _ = self.bert(x=aug_x, padding_masks=aug_padding_masks,g=aug_g, n_feat=aug_n_feat)  # (B, T, d_model)
        x2 = self.pooler(bert_output=x2, padding_masks=aug_padding_masks)  # (B, d_model)
        return x1, x2




class BERTContrastive_train():
    def __init__(self):
        train_dataset = GATDataset(x=train_trajectory_data, g=dgl_SH_map, n_feat=tmp_n_feat, aug_x=aug_train_trajectory_data, aug_g=aug_dgl_SH_map, aug_n_feat=aug_tmp_n_feat)
        valid_dataset = GATDataset(x=valid_trajectory_data, g=dgl_SH_map, n_feat=tmp_n_feat, aug_x=aug_valid_trajectory_data, aug_g=aug_dgl_SH_map, aug_n_feat=aug_tmp_n_feat)
        # test_dataset = GATDataset(x=test_trajectory_data, g=dgl_SH_map, n_feat=tmp_n_feat, aug_x=aug_test_trajectory_data, aug_g=aug_dgl_SH_map, aug_n_feat=aug_tmp_n_feat)
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                    collate_fn=collate,
                                    pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, 
                                    collate_fn=collate,
                                    pin_memory=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, 
        #                             collate_fn=collate,
        #                             pin_memory=True)

        config = {'key': 'value', 'pooling': 'cls'}
        data_feature = {'key': 'value'}
        self.model = BERTContrastive(config=config, data_feature=data_feature).cuda()
        # set learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        self.n_views = 2
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = 0.05
        self.device = 'cuda:0'

        self.min_dict = {}
        self.min_dict['min_con_loss'] = 1e18

    def _simclr_loss(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        # simclr loss
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)


        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature

        loss_res = self.criterion(logits, labels)
        return loss_res


    def train(self):
        self.model.train()
        iter = 0
        for batch in tqdm.tqdm(self.train_loader):
            x1, x2 = self.model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], 
                                batch['aug_x'], batch['aug_padding_masks'], batch['aug_g'], batch['aug_n_feat'])
            # print(x1.shape) // print(x2.shape)
            # # (B, d_model) // (B, d_model)
            # print(batch['x'])

            con_loss = self._simclr_loss(x1, x2)

            con_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Train con_Loss: {con_loss.item():.4f}")
            if ((iter + 1) % 100 == 0):
                valid_con_loss = self.valid()
                print('valid con_loss: ', valid_con_loss)
                if self.min_dict['min_con_loss'] > valid_con_loss:
                    self.min_dict['min_con_loss'] = valid_con_loss
                    if not os.path.exists('/nas/user/wyh/TNC/model/eta_data/'):
                        os.mkdir('/nas/user/wyh/TNC/model/eta_data/')
                    torch.save({
                        'model': self.model.state_dict(),
                        'best_loss': valid_con_loss,
                        'opt': self.optimizer,
                    }, '/nas/user/wyh/TNC/model/eta_data/con_model.pth.tar')

                self.model.train()
            if (iter + 1) % 100 == 0:
                print('train_con_loss: ', con_loss.item(), 'valid con_loss: ', self.min_dict['min_con_loss'])
            iter += 1


    def valid(self):
        with torch.no_grad():
            self.model.eval()
            avg_con_loss = 0
            avg_cnt = 0

            for batch in tqdm.tqdm(self.valid_loader):
                x1, x2 = self.model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], 
                                    batch['aug_x'], batch['aug_padding_masks'], batch['aug_g'], batch['aug_n_feat'])

                con_loss = self._simclr_loss(x1, x2)

                avg_con_loss += con_loss.item()
                avg_cnt += 1

            print ('valid con_loss: ', avg_con_loss / avg_cnt)
        return avg_con_loss / avg_cnt


    # def test(self):
    #     checkpoint = torch.load('/nas/user/wyh/TNC/model/eta_data/gat.model.pth.tar')
    #     self.eta_model.load_state_dict(checkpoint['model'])

    #     with torch.no_grad():
    #         self.eta_model.eval()
    #         avg_mape = 0
    #         avg_mae = 0
    #         avg_cnt = 0
    #         for i, batch in tqdm.tqdm(enumerate(self.test_loader)):
    #             mape, mae = self.eta_model(batch['x'], batch['padding_masks'], batch['g'], batch['n_feat'], batch['travel_time'])
    #         # for input in tqdm.tqdm(self.test_loader):
    #         #     mape, mae = self.eta_model(*input)

    #             # trick-1: 去除异常值
    #             if mape.item() > 2:
    #                 continue
    #             else:
    #                 avg_mape += mape.item()
    #                 avg_mae += mae.item()
    #                 avg_cnt += 1
    #             # avg_mape += mape.item()
    #             # avg_mae += mae.item()
    #             # avg_cnt += 1

    #             print('test mape: ', avg_mape / avg_cnt, ' test mae: ', avg_mae / avg_cnt)

    #     return avg_mape / avg_cnt, avg_mae / avg_cnt

if __name__ == '__main__':
    model_train = BERTContrastive_train()
    num_epochs = 20 # 训练轮数
    print('Training Start')
    for epoch in range(num_epochs):
        print ('epoch: ',epoch)
        model_train.train()

    # model_train.test()

