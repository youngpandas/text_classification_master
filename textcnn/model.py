import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from textcnn.config import TEMP_PATH

# num_embeddings = 5844
# num_classes = 10

# embedding_dim = 300  # 300
# num_kernel = 100  # 100
# kernel_sizes = [3, 4, 5]  # 3,4,5
# dropout = 0.5  # 0.5

args = {
    'num_words': 42725,
    'num_classes': 20,
    'embedding_dim': 300,
    'num_kernel': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5
}


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        num_words = args['num_words']
        num_classes = args['num_classes']

        embedding_dim = args['embedding_dim']  # 300
        num_kernel = args['num_kernel']  # 100
        kernel_sizes = args['kernel_sizes']  # 3,4,5
        dropout = args['dropout']  # 0.5
        mode = 'multi'

        word2vec = np.load(TEMP_PATH + '/word2vector.npy')
        word2vec = torch.from_numpy(word2vec).float()
        self.embeddings = nn.ModuleList([nn.Embedding.from_pretrained(word2vec, padding_idx=num_words, freeze=False)])
        if mode == 'multi':
            self.embeddings.append(nn.Embedding.from_pretrained(word2vec, padding_idx=num_words, freeze=True))

        self.convs = nn.ModuleList([nn.Conv2d(1, num_kernel, (k, embedding_dim * (2 if mode == 'multi' else 1)))
                                    for k in kernel_sizes])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_kernel * len(kernel_sizes), num_classes, bias=True)
        )

    def forward(self, x):
        # 如果dropout放在fc之后效果会特别差
        x = [embed(x) for embed in self.embeddings]
        x = torch.cat(x, 2)
        x = x.unsqueeze(1)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = TextCNN(args)
    x = torch.randint(0, 5000, (64, 300))
    y = model(x)
