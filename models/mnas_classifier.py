from torch import nn
import torchvision.models as models

from constants import DEVICE
from models.layers.mdconv import MDConv
from models.layers.sqex import SELayer, Swish
from models.layers.flatten import Flatten
from models.layers.activations import Elliot


class MnasXs(nn.Module):

    def __init__(
            self,
            input_size: int = 128
    ):
        super().__init__()

        self.name = "mnas_xs"
        self.input_size = input_size
        self.downsample_rate = 32
        self.size_before_classifier = self.input_size // self.downsample_rate

        mnasnet = models.mnasnet0_5(pretrained=True).to(DEVICE)
        self.mnas_head = mnasnet.layers[0:9]

        self.core = nn.Sequential(
            MDConv(in_channels=16, kernel_sizes=[3, 3, 7, 9], dilations=[1, 2, 1, 1], multiply_groups=1, stride=2),
            SELayer(16, 2),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflection'),
            MDConv(in_channels=16, kernel_sizes=[3, 5, 7, 9], dilations=[1, 1, 1, 1], multiply_groups=1, stride=2),
            SELayer(16, 2),
            nn.GroupNorm(4, 16)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(16, 12, 3, padding=1, padding_mode='reflection'),
            nn.MaxPool2d(2, 2),
            nn.GroupNorm(4, 12)
        )

        self.classifier = nn.Sequential(
            Swish(),
            Flatten(),
            nn.Linear(12 * self.size_before_classifier ** 2, 360),
            nn.Linear(360, 196, bias=True),
            Elliot()
        )

    def forward(self, image):
        x = self.mnas_head(image)
        x = self.core(x)
        x = self.conv(x)
        return self.classifier(x)
