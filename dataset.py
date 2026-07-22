import torch
from torch.utils.data import Dataset
import pandas as pd


class FredDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(
            csv_file,
            header=None
        )


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        # erste Spalte = Label
        label = int(row.iloc[0])

        # Rest = Pixel
        image = row.iloc[1:].values.astype("float32")

        # Tensor
        image = torch.tensor(image)

        label = torch.tensor(label)

        return image, label