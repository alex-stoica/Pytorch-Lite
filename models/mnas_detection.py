from torch import nn
import torch
from constants import LOAD_MODEL, DEVICE
from models.mnas_classifier import MnasXs
from models.layers.activations import Swish
from models.layers.flatten import Flatten, Flatten2


class MnasExtra(nn.Module):

    def __init__(
            self,
            input_size: int = 196,
            previous_input_size: int = 196
    ):
        super().__init__()

        self.name = "mnas_extra"
        self.input_size = input_size
        self.downsample_rate = 32
        self.size_before_classifier = self.input_size // self.downsample_rate

        mnas_core = MnasXs(input_size=previous_input_size).to(DEVICE)
        if LOAD_MODEL:
            mnas_core.load_state_dict(torch.load(LOAD_MODEL))
        self.core = nn.Sequential(mnas_core.mnas_head, mnas_core.core, mnas_core.conv)

        self.classifier = nn.Sequential(
            Swish(),
            Flatten(),
            nn.Linear(12 * self.size_before_classifier ** 2, 360),
            nn.Linear(360, 4, bias=True),
            Flatten2()
        )

    def forward(self, image):
        x = self.core(image)
        return self.classifier(x)
