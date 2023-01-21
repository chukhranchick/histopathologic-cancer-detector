import json

import numpy as np
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


def fetch_json(path: str) -> dict:
    with open(path, "r") as file:
        json_file = json.load(file)
        file.close()
    return json_file


def save_json(json_dict: dict, path: str):
    with open(path, "w") as file:
        json.dump(json_dict, file)
        file.close()
