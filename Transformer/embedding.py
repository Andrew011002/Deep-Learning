import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self, vocab_size, dm, pad_id, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dm, pad_id, **kwargs)

    def forward(self, x):
        # inshape: x - (batch_size, seq_len)
        return self.embedding(x)
    
    def unembedding(self):
        # unembed tensor shape: weight - (vocab_size, dm)
        return self.embedding.weight


if __name__ == '__main__':
    pass
