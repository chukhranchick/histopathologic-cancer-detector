import os

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class HistopathologicDataset(Dataset):
    def __init__(self, data_dir: str, labels: str, transformation: transforms):
        self.df = pd.read_csv(labels)
        self.df['id'] = self.df['id'].apply(lambda x: os.path.join(data_dir, x))
        self._transform = transformation

    def __len__(self):
        len(self.df)

    def __getitem__(self, item):
        image, label = self.df[item]
        with Image.open(image) as img:
            image = self.transform(img)
        return image, label

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
