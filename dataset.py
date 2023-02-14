import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm.auto import tqdm


class HistopathologicDataset(Dataset):
    def __init__(self, data_dir: str, labels: str, transformation: transforms):
        print("Loading data...")
        self.df = pd.read_csv(labels)
        print("Data loaded.")
        self.df['id'] = self.df['id'].apply(lambda x: os.path.join(data_dir, x))
        self._transform = transformation
        print('Converting images to tensors...')
        self.tensors = self.__images_to_tensor()
        print('Done')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        _, label = self.df.iloc[index]
        return self.tensors[index], label

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value

    def __images_to_tensor(self):
        tensors = []
        for i in tqdm(range(len(self.df))):
            image, label = self.df.iloc[i]
            with Image.open(image) as img:
                tensors.append(self.transform(img).detach())
        print("trying to stack tensor list in pytorch tensor...")
        return torch.stack(tensors)
