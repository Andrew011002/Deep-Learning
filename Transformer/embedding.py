import torch
import torch.nn as nn
import numpy as np


class Embeddings(nn.Module):

    def __init__(self, n_tokens, d_model, pad_idx, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model, pad_idx, **kwargs)

    def forward(self, x):
        # inshape: (batch_size, seq_len)
        return self.embedding(x)


if __name__ == '__main__':
    maxlen = 50
    vocab_size = 10000
    d_model = 512
    pad_idx = 0
    embed = Embeddings(vocab_size, d_model, pad_idx)
    sequences = torch.randint(0, vocab_size, (64, maxlen))
    embeddings = embed(sequences)
    print(embeddings.size())