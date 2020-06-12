from textrnn.config import TEMP_PATH

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        num_words = 42725
        num_classes = 20

        embedding_dim = 300
        hidden_size = 150

        word2vec = torch.from_numpy(np.load(TEMP_PATH + '/word2vector.npy')).float()
        self.embedding = nn.Embedding.from_pretrained(word2vec, padding_idx=num_words)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = torch.mean(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = LSTM()
    x = torch.randint(0, 5000, (64, 300))
    y = model(x)
    print(y.size())
