import torch.optim as optim
import torch.nn as nn
import torch
from Discriminator import Discriminator
from generator import gen
from torch.utils.data import Dataset, DataLoader
import numpy as np

x = np.load('false.npy')
train_data = x[:, :65]
y = x[:, -1:]
train_tensor = torch.tensor(train_data, dtype=torch.float32)
train_tensor = train_tensor.unsqueeze(1)
label_tensor = torch.tensor(y, dtype=torch.float32)


class CustomDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label


    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        sample = self.Data[idx]
        label = self.Label[idx]
        return sample, label


D = Discriminator()
G = gen()
d_optim = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(G.parameters(), lr=0.0001)
loss_fn = torch.nn.BCELoss()

dataset = CustomDataset(train_tensor,label_tensor)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

D_loss = []
G_loss = []

for epoch in range(20): #训练20个epoch
    d_epoch_loss = 0 # 初始损失值为0
    g_epoch_loss = 0
    # len(dataloader)返回批次数，len(dataset)返回样本数
    count = len(dataloader)
    for batch_idx, (data, target) in enumerate(dataloader):
        print(batch_idx)
        print(target)
        print(data)
        d_optim.zero_grad()
        print(batch_idx)
        real_output = D(data)

        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()

        random_signal = torch.randn(5, 1, 65)
        gen_signal = G(random_signal)
        fake_output = D(gen_signal.detach())
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()
        g_optim.zero_grad()
        fake_output = D(gen_signal)
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        # 计算平均的loss值
        d_epoch_loss /= count
        g_epoch_loss /= count
        # 将平均loss放入到loss数组中
        D_loss.append(d_epoch_loss.item())
        G_loss.append(g_epoch_loss.item())
        # 打印当前的epoch
        print('Epoch:', epoch)
        # 调用绘图函数







# 模型训练
def train_model():
    G.train()
    D.train()




'''def example():
    for epoch in range(100):  # 训练100个epoch
        optimizer.zero_grad()  # 梯度清零
        outputs = net(x)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()'''
