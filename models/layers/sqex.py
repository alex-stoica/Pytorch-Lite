import torch.nn as nn
from models.layers.activations import Swish


class SELayer(nn.Module):
    def __init__(self, channel, squeeze_ch):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, squeeze_ch, bias=False),
            Swish(),
            nn.Linear(squeeze_ch, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
