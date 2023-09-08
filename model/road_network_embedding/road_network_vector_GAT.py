import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

import sys
sys.path.append("/nas/user/wyh/TNC")
from pybind.test_funcs import test_map
from pybind.test_funcs import Map

# # 定义一个函数，用于计算图数据的入度和出度
def compute_degree_features(graph):
    # graph是一个networkx.Graph对象，表示图数据；返回一个torch.Tensor对象，shape为[n_nodes, 2]，表示每个节点的入度和出度特征向量
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    print(n_nodes)
    # 初始化特征矩阵
    features = torch.zeros(n_nodes, 2)
    # 遍历每个节点
    for i, node in enumerate(nodes):
        # 计算节点的入度和出度
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        # 将入度和出度的值归一化到[0, 1]区间
        in_degree = in_degree / (n_nodes - 1)
        out_degree = out_degree / (n_nodes - 1)
        # 将入度和出度的值赋给特征矩阵
        features[i, 0] = in_degree
        features[i, 1] = out_degree
    # 返回特征矩阵
    return features


raw_SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
vertex = raw_SH_map.v_section_mode_map()
edge = raw_SH_map.e_section_mode_map()

G = nx.DiGraph() # 创建一个有向图
G.add_nodes_from(vertex) # 从列表中添加多个节点
G.add_edges_from(edge) # 从列表中添加多条有向边

# 转换为numpy矩阵
adj = nx.to_numpy_matrix(G)
# 转换为torch张量
adj = torch.from_numpy(adj).float()

# 计算图的入度和出度特征
features = compute_degree_features(G)


# 将networkx图转换为PyTorch Geometric数据
data = from_networkx(G)
data.x = features
data.y = adj
# 打印数据信息
print(data)
# Data(edge_index=[2, 9295], num_nodes=4223, x=[4223, 2], y=[4223, 4223])
# Data(edge_index=[2, 4], x=[4, 2], y=[4])

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        linear = nn.Linear(128, 4223)
        # 将张量x送入线性层，得到[4223, 4223]维度的张量y
        x = linear(x)
        return x

# 创建模型对象
model = GAT(in_channels=2, out_channels=16, heads=8)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
model.train()
o = model(data)
# torch.Size([4223, 128])
print(o.size())
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')


# test_map()
print("end")
