import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.start1 = nn.Conv1d(in_channels=1, out_channels=32,kernel_size=3)
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.flat = nn.Flatten()
        self.connect1 = nn.Linear(192, 64)
        self.drop = nn.Dropout(0.5)
        self.output = nn.Linear(64, 1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,kernel_size=3)

    def forward(self, x):
        x = F.leaky_relu(self.start1(x))
        x = self.max_pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x)

        x = self.connect1(x)
        x = F.leaky_relu(self.drop(x))
        x = self.output(x)

        if x > 0:
            x = 1
        else:
            x = 0

        return x
