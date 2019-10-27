import torch
from torch import nn
from typing import List


def _split_channels(total_filters, num_groups):
    """
    https://github.com/tensorflow/tpu/blob/d9b1c2bda56167cd3e61d2087ff1853aa84e68cc/models/official/mnasnet/mixnet/custom_layers.py#L33
    """
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


class MDConv(nn.Module):
    # MDConv(300,[3,5,7,9])

    def __init__(
            self,
            in_channels: int,
            kernel_sizes: List,
            dilations: List,
            multiply_groups: int = 2,
            depthwise: bool = True,
            padding_mode='reflection',
            stride: int = 1
    ):
        super().__init__()

        # [3,5,7], multiply_groups = 2 => [3,3,5,5,7,7]
        # if multiply_groups != 1:
        #     print(multiply_groups, "ksize_", kernel_sizes)
        #     kernel_sizes = [kernel_sizes[i // 2] for i in range(len(kernel_sizes) * 2)]

        self.in_channels = _split_channels(in_channels, len(kernel_sizes))

        self.convs = nn.ModuleList()

        for ch, k, d in zip(self.in_channels, kernel_sizes, dilations):
            pad = (k - 1 + d) // 2

            if depthwise:
                groups = ch
            else:
                groups = 1

            self.convs.append(
                nn.Conv2d(ch, ch, k, stride=stride, padding=pad, groups=groups, dilation=d, padding_mode=padding_mode))

    def forward(self, x):
        groups = torch.split(x, self.in_channels, 1)
        return torch.cat([conv(x) for conv, x in zip(self.convs, groups)], 1)
