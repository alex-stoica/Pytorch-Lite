import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Flatten2(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 2, -1)
