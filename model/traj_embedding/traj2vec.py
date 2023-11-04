# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 设置设备为CPU或GPU，如果有CUDA可用的话
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 定义LSTM模型类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 通过LSTM层得到输出和最终的隐藏状态和细胞状态
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出，通过全连接层得到预测值
        out = self.fc(out[:, -1, :])
        return out
    


# 设置超参数
input_size = 1 # 输入特征维度，例如经度、纬度、速度、方向
output_size = 1 # 输出特征维度，例如行程时间
hidden_size = 64 # 隐藏层大小
num_layers = 2 # LSTM层数
num_epochs = 8 # 训练轮数
batch_size = 32 # 批次大小
learning_rate = 0.01 # 学习率


# 加载轨迹数据集，假设已经分好了训练集和测试集，并保存为csv文件
train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_valid.csv", sep=';', header=0)
test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_test.csv", sep=';', header=0)

# 将字符串转换为数字列表，并计算每个序列的长度
train_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:,1].values]
train_time_data = [list(map(int, x.strip('[]').split(','))) for x in train_data.iloc[:, 2].values]
# 有-999的情况出现
train_times_data = []
for x in train_time_data:
    if len(x)>128:
        x = x[:128]
    if x[0] == -999:
        train_times_data.append(0)
    elif x[-1]== -999:
        train_times_data.append(x[-2]-x[0])
    else:
        train_times_data.append(x[-1]-x[0])

lengths = [len(x) for x in train_data.iloc[:,1].values]
# 使用-999填充到相同长度，并转换为Tensor
max_len = max(lengths)
train_trajectory_data = [x + [-999] * (max_len - len(x)) for x in train_trajectory_data]
# 归一化train_times_data
train_times_data = np.array(train_times_data)
train_times_data = (train_times_data - train_times_data.min()) / (train_times_data.max() - train_times_data.min())
train_times_data = train_times_data.tolist()
# print(train_times_data)
# # 断点语句
# l=1
# assert l==0, "error"


X_train = torch.tensor(train_trajectory_data).float()
X_train = torch.narrow(X_train, dim=1, start=0, length=128).to(device)
y_train = torch.tensor(train_times_data).float().to(device)
# print(y_train)

# 将字符串转换为数字列表，并计算每个序列的长度
valid_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:,1].values]
valid_time_data = [list(map(int, x.strip('[]').split(','))) for x in valid_data.iloc[:, 2].values]
valid_times_data = []
for x in valid_time_data:
    if len(x)>128:
        x = x[:128]
    if x[0] == -999:
        valid_times_data.append(0)
    elif x[-1]== -999:
        valid_times_data.append(x[-2]-x[0])
    else:
        valid_times_data.append(x[-1]-x[0])
lengths = [len(x) for x in valid_data.iloc[:,1].values]
# 使用-999填充到相同长度，并转换为Tensor
max_len = max(lengths)
valid_trajectory_data = [x + [-999] * (max_len - len(x)) for x in valid_trajectory_data]
# 归一化valid_times_data
valid_times_data = np.array(valid_times_data)
valid_times_data = (valid_times_data - valid_times_data.min()) / (valid_times_data.max() - valid_times_data.min())
valid_times_data = valid_times_data.tolist()

X_valid = torch.tensor(valid_trajectory_data).float()
X_valid = torch.narrow(X_valid, dim=1, start=0, length=128).to(device)
y_valid = torch.tensor(valid_times_data).float().to(device)

# 将字符串转换为数字列表，并计算每个序列的长度
test_trajectory_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:,1].values]
test_time_data = [list(map(int, x.strip('[]').split(','))) for x in test_data.iloc[:, 2].values]
test_times_data = []
for x in test_time_data:
    if len(x)>128:
        x = x[:128]
    if x[0] == -999:
        test_times_data.append(0)
    elif x[-1]== -999:
        test_times_data.append(x[-2]-x[0])
    else:
        test_times_data.append(x[-1]-x[0])
lengths = [len(x) for x in test_data.iloc[:,1].values]
# 使用-999填充到相同长度，并转换为Tensor
max_len = max(lengths)
test_trajectory_data = [x + [-999] * (max_len - len(x)) for x in test_trajectory_data]
# 归一化test_times_data
test_times_data = np.array(test_times_data)
test_times_data = (test_times_data - test_times_data.min()) / (test_times_data.max() - test_times_data.min())
test_times_data = test_times_data.tolist()


X_test = torch.tensor(test_trajectory_data).float()
X_test = torch.narrow(X_test, dim=1, start=0, length=128).to(device)
y_test = torch.tensor(test_times_data).float().to(device)

# 将数据转换为张量，并划分为输入和输出
# X_train = torch.tensor(train_data.iloc[:,1].values).float().to(device)
# y_train = torch.tensor(train_data.iloc[:, 2].values).float().to(device)
# X_valid = torch.tensor(valid_data.iloc[:, 1].values).float().to(device)
# y_valid = torch.tensor(valid_data.iloc[:, 2].values).float().to(device)
# X_test = torch.tensor(test_data.iloc[:, 1].values).float().to(device)
# y_test = torch.tensor(test_data.iloc[:, 2].values).float().to(device)

print("dataload start!")

# 创建数据加载器，以便进行批次训练
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)

# 创建LSTM模型实例，并将其移动到相应的设备上
model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)

# 定义损失函数和优化器，这里使用均方误差作为损失函数，使用Adam作为优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(loader):
    with torch.no_grad():
        total_loss = 0
        total_count = 0
        for inputs, targets in tqdm.tqdm(loader):
            inputs = inputs.reshape(-1, inputs.size(1), input_size)
            targets = targets.reshape(-1, output_size)
            # print(targets)
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)
        return total_loss / total_count
    
# 训练模型
best_valid_loss = float("inf") # 初始化最佳的验证损失值为无穷大
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(tqdm.tqdm(train_loader)):
        # 将输入和目标调整为正确的形状，即（批次大小，序列长度，特征维度）
        inputs = inputs.reshape(-1, inputs.size(1), input_size)
        # print(inputs)
        targets = targets.reshape(-1, output_size)
        # print(targets)

        # 前向传播
        outputs = model(inputs)
        # print(outputs)
        # assert 2==1,"stop"
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在验证集上评估模型的性能，并保存最佳的模型参数
    valid_loss = evaluate(valid_loader)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "/nas/user/wyh/TNC/data/ckpt/best_model.ckpt")
    # 打印每轮的训练损失和验证损失
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Valid Loss: {valid_loss:.4f}")

# 加载最佳的模型参数，并在测试集上评估模型的性能
model.load_state_dict(torch.load("/nas/user/wyh/TNC/data/ckpt/best_model.ckpt"))
test_loss = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}")
