from torch.utils.data import Dataset
from textcnn.config import TEMP_PATH
import numpy as np
from torch import tensor
import torch


class MyDataset(Dataset):

    def __init__(self, cate, seq_len):
        self.x = np.load(TEMP_PATH + "/{}.input.npy".format(cate))
        self.y = np.load(TEMP_PATH + "/{}.target.npy".format(cate))
        if seq_len > len(self.x[0]):
            assert "wrong"
        self.x = [e[:seq_len] for e in self.x]

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x, y = tensor(x, dtype=torch.long), tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)
