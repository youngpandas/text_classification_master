import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x).sum(1)
        return x


class HAN(nn.Module):

    def __init__(self):
        super(HAN, self).__init__()
        num_embeddings = 27611
        num_classes = 20
        num_sentences = 15
        num_words = 15

        embedding_dim = 200  # 200
        hidden_size_gru = 50  # 50
        hidden_size_att = 100  # 100

        self.num_words = num_words
        self.embed = nn.Embedding(num_embeddings + 1, embedding_dim, num_embeddings)

        self.gru1 = nn.GRU(embedding_dim, hidden_size_gru, bidirectional=True, batch_first=True)
        self.att1 = SelfAttention(hidden_size_gru * 2, hidden_size_att)

        self.gru2 = nn.GRU(hidden_size_att, hidden_size_gru, bidirectional=True, batch_first=True)
        self.att2 = SelfAttention(hidden_size_gru * 2, hidden_size_att)

        # 这里fc的参数很少，不需要dropout
        self.fc = nn.Linear(hidden_size_att, num_classes, True)

    def forward(self, x):
        # 64 512 200
        x = x.view(x.size(0) * self.num_words, -1).contiguous()
        x = self.embed(x)
        x, _ = self.gru1(x)
        x = self.att1(x)
        x = x.view(x.size(0) // self.num_words, self.num_words, -1).contiguous()
        x, _ = self.gru2(x)
        x = self.att2(x)
        x = self.fc(x)
        return x
