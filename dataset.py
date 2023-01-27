import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class HistopathologicDataset(Dataset):
    def __init__(self, data_dir: str, labels: str, transformation: transforms):
        self.df = pd.read_csv(labels)
        self.df['id'] = self.df['id'].apply(lambda x: os.path.join(data_dir, x))
        self._transform = transformation

    def __len__(self):
        len(self.df)

    def __getitem__(self, index: int):
        image, label = self.df.iloc[index]
        with Image.open(image) as img:
            image = self.transform(img)
        return image, label

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
