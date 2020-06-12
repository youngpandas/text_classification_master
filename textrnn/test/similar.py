import torch
from torch import nn
import numpy as np

from textrnn.config import TEMP_PATH, RECORD_PATH

import os


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        num_words = 42725
        num_classes = 20
        embedding_dim = 300
        hidden_size = 150

        self.embedding = nn.Embedding(num_words + 1, embedding_dim, padding_idx=num_words)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_size, num_classes)
        )


if __name__ == '__main__':
    model = LSTM()
    model.load_state_dict(torch.load(RECORD_PATH + "/LSTM.769.pth"))
