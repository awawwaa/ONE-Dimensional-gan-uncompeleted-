import numpy as np
import torch

from gan import G_net,G_net2,Generator1D,G_net3,CustomNet
import torch.nn as nn
import matplotlib.pyplot as plt
from Discriminator import Discriminator
from generator import gen


G = gen()
D = Discriminator()
random_noise = torch.randn(1, 1, 64)
noise = G(random_noise)
OUT = D(noise)
print(OUT)
print(noise)