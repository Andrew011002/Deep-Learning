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

        # calc context & attn then add & norm (residual connections) shape: (batch_size, seq_len, dm)
        x = src
        x_out, attn = self.multihead(x, x, x, mask=src_mask)
        x = self.norm1(x + x_out)

        # calc linear transforms then add & norm (residual connections) shape: (batch_size, seq_len, dm)
        x_out = self.feedforward(x)
        out = self.norm2(x + x_out)
        return out, attn

class Decoder(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.maskmultihead = MultiHeadAttention(dm, nhead, bias, dropout)
        self.multihead = MultiHeadAttention(dm, nhead, bias, dropout)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout)
        self.norm1 = Norm(dm, eps)
        self.norm2 = Norm(dm, eps)
        self.norm3 = Norm(dm, eps)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape src: (batch_size seq_len, dm) inshape tgt: (batch_size, seq_len - 1, dm)

        # calc masked context & attn then add & norm (residual connections) shape: (batch_size, seq_len - 1, dm)
        x = tgt
        x_out, attn1 = self.maskmultihead(x, x, x, mask=tgt_mask)
        x = self.norm1(x + x_out)

        # calc context & attn then add & norm (residual connections) shape: (batch_size, seq_len - 1, dm)
        x_out, attn2 = self.multihead(x, src, src, mask=src_mask)
        x = self.norm2(x + x_out)

        # calc linear transforms then add & norm (residual connections) shape: (batch_size, seq_len - 1, dm)
        x_out = self.feedforward(x)
        out = self.norm3(x + x_out)
        return out, attn1, attn2



if __name__ == "__main__":
    pass


