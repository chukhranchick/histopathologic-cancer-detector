import os

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class HistopathologicCancerDS(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 transform=None):
        self.transform = transform
        processed_labels_path: str = os.path.join('data', 'labels.pt')
        processed_images_path = os.path.join('data', 'images64.pt')
        if os.path.exists(processed_labels_path) and os.path.exists(processed_images_path):
            self.images = torch.load(processed_images_path)
            self.labels = torch.load(processed_labels_path)
        else:
            self.labels = dataframe['label'].to_numpy()
            self.labels = torch.tensor(self.labels)
            self.abs_image_path = dataframe['id'].to_numpy()
            print("Converting images to tensors...")
            self.images = self._images_to_tensors()
            print("Trying to save tensors to files...")
            if not os.path.exists('data'):
                os.mkdir('data')
            torch.save(self.images, processed_images_path)
            torch.save(self.labels, processed_labels_path)
        print("Done")
        print("The dataset is ready for use.")

    def __len__(self):
        return len(self.labels)

    def _images_to_tensors(self):
        images = []
        for i in tqdm(range(len(self.abs_image_path))):
            with Image.open(self.abs_image_path[i]) as image:
                images.append(self.transform(image))
        return torch.stack(images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class ProcessedHistopathologicCancerDS(Dataset):
    def __init__(self, data_path: str):
        self.images = torch.load(os.path.join(data_path, 'images64.pt'))
        self.labels = torch.load(os.path.join(data_path, 'labels.pt'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
