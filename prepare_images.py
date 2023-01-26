import os

import numpy as np
import pandas as pd


def prepare_data(labels_dir: str, set_type: str = 'train') -> None:
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
    dataframe.set_index('id').to_csv("processed_labels.csv")


def main() -> None:
    LABELS_DIR: str = 'train_labels.csv'
    TRAIN_DATA: str = 'train'
    prepare_data(LABELS_DIR, TRAIN_DATA)


if __name__ == '__main__':
    main()
