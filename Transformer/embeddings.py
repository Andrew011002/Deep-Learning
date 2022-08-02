import torch
import torch.nn as nn
import numpy as np


class Embeddings(nn.Module):

    def __init__(self, n_tokens, d_model, pad_idx, **kwargs) -> None:
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model, pad_idx, kwargs)

    def forward(self, x):
        return self.embedding(x)


if __name__ == '__main__':
    pass