import torch
import torch.nn as nn
import torch.nn.functional as F


class gen(nn.Module):
    def __init__(self):
        super(gen,self).__init__()
        self.activate1 = nn.LeakyReLU()
        self.cov = nn.Conv1d(in_channels= 1,out_channels=32,kernel_size=3)
        self.cov2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3)
        self.cov3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.output = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=5)
        self.connect = nn.Linear(in_features=54,out_features=64)

    def forward(self,x):
        x = F.leaky_relu(x)
        x = self.cov(x)
        x = F.leaky_relu(x)
        x = self.cov2(x)
        x = F.leaky_relu(x)
        x = self.cov3(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        x = self.connect(x)

        return x
