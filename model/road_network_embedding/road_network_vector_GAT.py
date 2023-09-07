import torch
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
# # 打印结果
# print(features)
# print(features.size())

# 创建一个有向图，每个节点有两个特征：x和y
G = nx.DiGraph()
G.add_nodes_from([
    (0, {'x': 0.1, 'y': 0.2}),
    (1, {'x': 0.3, 'y': 0.4}),
    (2, {'x': 0.5, 'y': 0.6}),
    (3, {'x': 0.7, 'y': 0.8}),
])
G.add_edges_from([
    (0, 1),
    (1, 2),
    (2, 0),
    (2, 3),
])

# 将networkx图转换为PyTorch Geometric数据
data = from_networkx(G)
# 将节点特征转换为张量
data.x = torch.tensor([[node['x'], node['y']] for node in G.nodes.values()], dtype=torch.float)
# 将节点标签转换为张量（这里假设每个节点的标签就是它的索引）
data.y = torch.tensor([node for node in G.nodes()], dtype=torch.long)
# 打印数据信息
print(data)
# Data(edge_index=[2, 4], x=[4, 2], y=[4])

# 定义GAT模型，输入特征维度为2，输出特征维度为16，头数为8，分类层维度为4
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(2, 16, heads=8)
        self.conv2 = GATConv(16 * 8, 4)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

# 创建模型实例
model = GAT()
# 定义损失函数和优化器
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.y], data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    pred = model(data).max(dim=1)[1]
    acc = pred.eq(data.y).sum().item() / data.num_nodes
    print(f'Accuracy: {acc:.4f}')


# test_map()
print("end")
