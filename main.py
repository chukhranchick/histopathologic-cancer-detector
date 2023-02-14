import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from dataset import HistopathologicDataset
from models import Conv4Net, init_weight
from utils.train import train
from utils.constants import SETTINGS_FILE
from utils.json import fetch_json, save_json


def main(file_config: dict, model_config: dict, result_file: str) -> None:
    TRAIN_BATCH_SIZE = model_config['train_batch_size']
    VALIDATION_BATCH_SIZE = model_config['validation_batch_size']

    # Set only if you don't have memory enough to work with the batch
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = HistopathologicDataset(file_config['train_data'], file_config['processed_labels'], transform)
    print(f'Dataset length: {len(ds)}')

    train_size = int(0.9 * len(ds))
    validation_size = len(ds) - train_size
    train_ds, validation_ds = random_split(ds, [train_size, validation_size])

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)
    dataloaders = {
        'train': train_dl,
        'validation': validation_dl
    }

    models = [Conv4Net(model_config)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCEWithLogitsLoss()

    result_dir = file_config['results_dir']
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    full_result_path = os.path.join(result_dir, result_file)
    if not os.path.exists(full_result_path):
        save_json({}, full_result_path)

    for model in models:
        model = model.to(device)
        model.apply(init_weight)
        optimizer = Adam(model.parameters(), lr=5e-4)
        _stats = {
            'train': {
                'loss': [],
                'accuracy': []
            },
            'validation': {
                'loss': [],
                'accuracy': []
            }
        }
        model_name = model.__class__.__name__
        print(f"Training model: {model_name}")
        train(model, dataloaders, criterion, optimizer, epochs=100, stats=_stats, device=device, patience=10)
        print("loading previous training results...")
        all_stats = fetch_json(full_result_path)
        all_stats[model_name] = _stats
        print("saving current training results...")
        save_json(all_stats, full_result_path)


def show_stats(statistics: dict):
    for model in statistics.keys():
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(statistics[model]['train']['loss'], label='train')
        ax1.plot(statistics[model]['validation']['loss'], label='validation')
        ax2.plot(statistics[model]['train']['accuracy'], label='train')
        ax2.plot(statistics[model]['validation']['accuracy'], label='validation')
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        plt.title(model)
        plt.legend()
        plt.show()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for model in statistics.keys():
        ax1.plot(statistics[model]['train']['loss'], label=model)
        ax2.plot(statistics[model]['train']['accuracy'], label=model)
        ax3.plot(statistics[model]['validation']['loss'], label=model)
        ax4.plot(statistics[model]['validation']['accuracy'], label=model)
    ax1.set_title('Train Loss')
    ax2.set_title('Train accuracy')
    ax3.set_title('Validation Loss')
    ax4.set_title("Validation accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # letters = 'abcdefghijklmnopqrstuvwxyz'
    # upper_letters = letters.upper()
    # digits = '0123456789'
    # all_chars = letters + upper_letters + digits
    file_config = fetch_json(SETTINGS_FILE)['file_config']
    # model_config = fetch_json(SETTINGS_FILE)['model_config']
    # result_file = f"result_{''.join(np.random.choice(list(all_chars), 10))}.json"
    full_result_path = os.path.join(file_config['results_dir'], "result_JyOvRDpPBq.json")
    # main(file_config, model_config, result_file)
    stats = fetch_json(full_result_path)
    show_stats(stats)
