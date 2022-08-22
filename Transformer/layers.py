import torch
import torch.nn as nn
from sublayers import MultiHeadAttention, Norm, FeedForwardNetwork


class Encoder(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(dm, nhead, bias, dropout)
        self.norm1 = Norm(dm, eps)
        self.norm2 = Norm(dm, eps)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout)

    def forward(self, src, src_mask=None):
        # inshape: (batch_size, seq_len, dm)

        # calc attn then add & norm (residual connection) shape: (batch_size, seq_len, dm)
        x, attn = self.multihead(src, src, src, mask=src_mask)
        x = self.norm1(src + x)

        # calc linear transforms then add & norm (residual connections) shape: (batch_size, seq_len, dm)
        out = self.norm2(x + self.feedforward(x))
        return out

if __name__ == "__main__":
    inputs = torch.rand(32, 20, 512)
    encoder = Encoder(512, 8, 2048)
    outputs = encoder(inputs, src_mask=None)
    print(outputs.size())