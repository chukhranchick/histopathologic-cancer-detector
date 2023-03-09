from typing import Union

import numpy as np
import torch
from torch import nn


def eval_shape(height, width, model, pool=0):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            if isinstance(layer, nn.Conv2d):
                height = np.floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
                width = np.floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                height = np.floor((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
                width = np.floor((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    if pool:
        height /= pool
        width /= pool
    return int(height), int(width)


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1Net(nn.Module):
    def __init__(self, params: dict[str, Union[int, tuple[int, int]]]):
        super(Conv1Net, self).__init__()
        input_channels: int = params['in_channels']
        out_channels: int = params['out_channels']
        image_size: tuple = params['image_size']
        self.conv = ConvBlock(input_channels, out_channels, 7)
        self.max_pool = nn.MaxPool2d(3)
        h, w = eval_shape(*image_size, nn.Sequential(self.conv, self.max_pool))
        self.fc1 = nn.Linear(out_channels * h * w, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Conv2Net(nn.Module):
    def __init__(self, params: dict[str, Union[int, tuple[int, int]]]):
        super(Conv2Net, self).__init__()
        input_channels: int = params['in_channels']
        out_channels: int = params['out_channels']
        image_size: tuple = params['image_size']

        self.conv_block = nn.Sequential(
            ConvBlock(input_channels, out_channels, 7),
            nn.MaxPool2d(3),
            ConvBlock(out_channels, out_channels * 2, 5),
            nn.MaxPool2d(3)
        )
        h, w = eval_shape(*image_size, self.conv_block)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_channels * 2 * h * w, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Conv4Net(nn.Module):
    def __init__(self, params: dict[str, Union[int, tuple[int, int]]]):
        super(Conv4Net, self).__init__()
        input_channels: int = params['in_channels']
        out_channels: int = params['out_channels']
        image_size: tuple = params['input_image_size']

        self.block = nn.Sequential(
            ConvBlock(input_channels, out_channels, 7),
            nn.MaxPool2d(2),
            ConvBlock(out_channels, out_channels * 2, 5),
            nn.MaxPool2d(2),
            ConvBlock(out_channels * 2, out_channels * 4),
            nn.MaxPool2d(2),
            ConvBlock(out_channels * 4, out_channels * 8),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.relu = nn.ReLU()
        h, w = eval_shape(*image_size, self.block)
        self.fc1 = nn.Linear(out_channels * 8 * h * w, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
