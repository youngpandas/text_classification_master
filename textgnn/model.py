from torch import nn, tensor
import torch.nn.functional as F
import numpy as np
import torch


class GraphUint(nn.Module):
    def __init__(self):
        super(GraphUint, self).__init__()
        num_words = 5000

        self.word_embedding = nn.Embedding(num_words + 1, 300, num_words)
        self.edge_embedding = nn.Embedding(num_words * num_words + 1, 300, num_words)

    def forward(self, x):
        pass


class TextLevelGNN(nn.Module):

    def __init__(self):
        super(TextLevelGNN, self).__init__()
        num_nodes = 4904
        embedding_dim = 300
        num_classes = 14

        self.R = nn.Embedding(num_nodes + 1, embedding_dim, padding_idx=0)
        self.E = nn.Embedding(num_nodes * num_nodes + 1, 1, padding_idx=0)
        self.N = nn.Embedding(num_nodes + 1, 1, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, num_classes, bias=True),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1),
            nn.Dropout(0.5)
        )

    def forward(self, master_nodes, slave_nodes_list, slave_edges_list):
        Rn = self.R(master_nodes)
        Ra = self.R(slave_nodes_list)
        Ean = self.E(slave_edges_list)
        Mn = (Ra * Ean).max(dim=2)[0]
        Nn = self.N(master_nodes)
        x = (1 - Nn) * Mn + Nn * Rn
        x = self.fc(x.sum(dim=1))
        return x


if __name__ == '__main__':
    num_nodes = 4904
    batch_size = 64
    seq_len = 1000
    window_size = 2
    embedding_dim = 300
    num_classes = 10
    master_nodes = tensor(np.random.randint(0, num_nodes + 1, (batch_size, seq_len)), dtype=torch.long)
    slave_nodes_list = tensor(np.random.randint(0, num_nodes + 1, (batch_size, seq_len, window_size * 2)),
                              dtype=torch.long)
    slave_edges_list = torch.randint(0, num_nodes * num_nodes + 1, (batch_size, seq_len, window_size * 2))

    model = TextLevelGNN()
    y = model(master_nodes, slave_nodes_list, slave_edges_list)
    print(y.shape)
