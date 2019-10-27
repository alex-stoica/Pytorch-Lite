import torch.nn as nn
import torch


class Elliot(nn.Module):
    def forward(self, x):
        return 0.5 * x / (1 + abs(x)) + 0.5


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
