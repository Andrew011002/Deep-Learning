import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self, n_tokens, dm, pad_id, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, dm, pad_id, **kwargs)

    def forward(self, x):
        # inshape: x - (batch_size, seq_len)
        return self.embedding(x)
    
    def unembedding(self):
        # weight matrix for unembedding
        return self.embedding.weight


if __name__ == '__main__':
    pass
