import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from dataset import HistopathologicCancerDS


def prepare_data(labels_dir: str,
                 transformation: transforms.Compose,
                 set_type: str = 'train') -> None:
    dataframe: pd.DataFrame = pd.read_csv(labels_dir)
    dataframe = dataframe.drop_duplicates(subset=['id'], keep=False)
    normal_dataframe: pd.DataFrame = dataframe[dataframe['label'] == 0]
    malignant_dataframe: pd.DataFrame = dataframe[dataframe['label'] == 1]
    balance_length = min(len(normal_dataframe), len(malignant_dataframe))
    normal_dataframe = normal_dataframe[:balance_length]
    malignant_dataframe = malignant_dataframe[:balance_length]
    dataframe = pd.concat([normal_dataframe, malignant_dataframe])

    image_names: list[str] = sorted(os.listdir(set_type))
    dataframe['id'] = dataframe['id'].apply(lambda x: x + '.tif')
    intersected = np.intersect1d(dataframe['id'].to_numpy().tolist(), image_names)

    dataframe = dataframe[dataframe['id'].isin(intersected)]
    dataframe['id'] = dataframe['id'].apply(lambda x: os.path.join(set_type, x))
    image_names = dataframe['id'].to_numpy().tolist()
    ds = HistopathologicCancerDS(dataframe, 'images64tmp.pt', 'labelstmp.pt', transformation)
    # TODO remove the code below if won't be needed
    # labels = dataframe['label'].to_numpy().tolist()
    # images = []
    # for i in tqdm(range(len(image_names))):
    #     with Image.open(image_names[i]) as image:
    #         images.append(transformation(image))
    # images, labels = torch.stack(images), torch.tensor(labels)


def main() -> None:
    LABELS_DIR: str = 'train_labels.csv'
    TRAIN_DATA: str = 'train'
    CENTER_SIZE = 64
    transformation = transforms.Compose([transforms.CenterCrop(CENTER_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    prepare_data(LABELS_DIR, transformation, TRAIN_DATA)


if __name__ == '__main__':
    main()
