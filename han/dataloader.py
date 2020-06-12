from torch.utils.data import Dataset
import numpy as np
from torch import tensor
import torch

from han.config import TEMP_PATH


class MyDataset(Dataset):

    def __init__(self, cate, seq_len):
        self.x = np.load(TEMP_PATH + "/{}.x.npy".format(cate))
        self.y = np.load(TEMP_PATH + "/{}.y.npy".format(cate))
        if seq_len > len(self.x[0]):
            assert "wrong"
        self.x = [e[:seq_len] for e in self.x]

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x, y = tensor(x, dtype=torch.long), tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)
