import datetime
import gc
import os
from typing import Optional, Union

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np


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
        print(height, width)
    if pool:
        height /= pool
        width /= pool
    return int(height), int(width)


def one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer=None,
        device: str = 'cpu'):
    running_loss = 0.0
    running_accuracy = 0.0
    number_of_samples = 0
    for i, data in enumerate(tqdm(dataloader, desc='Batches', position=0, leave=False)):
        x, y = data
        x = x.half().to(device)
        y = y.to(device).float()

        sample_size = x.size(0)

        if optimizer:
            optimizer.zero_grad()

        with autocast():
            output = model(x).view(len(x))
            loss = criterion(output, y)
        output = torch.sigmoid(output)
        y = y.int()
        output = (output >= 0.5) * 1

        running_accuracy += torch.sum(output == y).item()
        running_loss += float(loss.item() * sample_size)
        number_of_samples += sample_size

        if optimizer:
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            del x, output, loss
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    loss = running_loss / len(dataloader.dataset)
    accuracy = running_accuracy / number_of_samples
    return loss, accuracy


def train(
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 50,
        stats: dict = None,
        device: str = 'cpu',
        early_stopping: bool = True,
        patience: int = 5) -> None:
    current_date = datetime.datetime.now().strftime('%d.%m.%y')
    model_name = model.__class__.__name__
    save_dir = os.path.join('weights', current_date, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, f'{model_name}.pt')
    early_stopping_counter: int = 0
    for epoch in range(epochs):
        print(f"epoch {epoch}/{epochs}")
        avg_train_loss, avg_train_accuracy = one_epoch(model, dataloaders['train'], criterion, optimizer, device)

        with torch.no_grad():
            avg_valid_loss, avg_valid_accuracy = one_epoch(model, dataloaders['validation'], criterion, device=device)

        if early_stopping:
            if stats['validation']['loss'] and stats['train']['loss']:
                if (avg_valid_loss > stats['validation']['loss'][-(1 + early_stopping_counter)] or
                        avg_train_loss > stats['train']['loss'][-(1 + early_stopping_counter)]):
                    early_stopping_counter += 1
                    if early_stopping_counter == patience:
                        print("Early Stopping...")
                        return
                else:
                    early_stopping_counter = 0

        if stats['validation']['loss'] and stats['train']['loss']:
            if (avg_valid_loss < stats['validation']['loss'][-(1 + early_stopping_counter)] or
                    avg_train_loss < stats['train']['loss'][-(1 + early_stopping_counter)]):
                save_state(model, optimizer, save_dir)

        stats['train']['loss'].append(avg_train_loss)
        stats['train']['accuracy'].append(avg_train_accuracy)

        stats['validation']['loss'].append(avg_valid_loss)
        stats['validation']['accuracy'].append(avg_valid_accuracy)

        print(f"\nTraining average loss - {avg_train_loss}, average accuracy - {avg_train_accuracy}")
        print(f"Validation loss - {avg_valid_loss},  accuracy - {avg_valid_accuracy}")


def save_state(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               filename: str) -> None:
    state_dict: dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state_dict, filename)


def load_state(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               filename: str) -> tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return model, optimizer


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
        image_size: tuple = params['image_size']

        self.block = nn.Sequential(
            ConvBlock(input_channels, out_channels, 7),
            nn.MaxPool2d(3),
            ConvBlock(out_channels, out_channels * 2, 5),
            nn.MaxPool2d(3),
            ConvBlock(out_channels * 2, out_channels * 4, 5),
            nn.MaxPool2d(2),
            ConvBlock(out_channels * 4, out_channels * 8),
            nn.MaxPool2d(2)
        )
        self.relu = nn.ReLU()
        h, w = eval_shape(*image_size, self.block)
        self.fc1 = nn.Linear(out_channels * 8 * h * w, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
