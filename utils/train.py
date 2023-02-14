import gc
import datetime
import os
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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
