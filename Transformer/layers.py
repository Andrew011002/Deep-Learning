import torch
import torch.nn as nn
from sublayers import MultiHeadAttention, Norm, FeedForwardNetwork


class EncoderLayer(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-5) -> None:
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

class Encoder(nn.Module):

    def __init__(self, dm, nhead, dff, layers=6, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.stack = nn.ModuleList([EncoderLayer(dm, nhead, dff, bias, dropout, eps) \
            for l in range(layers)])

    def forward(self, src, src_mask=None):
        # inshape src: (batch_size, seq_len, d_model)
        x = src
        # pass src & through stack of encoders (out of layer is in for next)
        for encoder in self.stack:
            x, attn = encoder(x, src_mask=src_mask)
        out = x
        return out, attn


class DecoderLayer(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.maskmultihead = MultiHeadAttention(dm, nhead, bias, dropout)
        self.multihead = MultiHeadAttention(dm, nhead, bias, dropout)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout)
        self.norm1 = Norm(dm, eps)
        self.norm2 = Norm(dm, eps)
        self.norm3 = Norm(dm, eps)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape src: (batch_size seq_len, dm) inshape tgt: (batch_size, tgt_len, dm)

        # calc masked context & attn then add & norm (residual connections) shape: (batch_size, tgt_len, dm)
        x = tgt
        x_out, attn1 = self.maskmultihead(x, x, x, mask=tgt_mask)
        x = self.norm1(x + x_out)

        # calc context & attn then add & norm (residual connections) shape: (batch_size, tgt_len, dm)
        x_out, attn2 = self.multihead(x, src, src, mask=src_mask)
        x = self.norm2(x + x_out)

        # calc linear transforms then add & norm (residual connections) shape: (batch_size, tgt_len, dm)
        x_out = self.feedforward(x)
        out = self.norm3(x + x_out)
        return out, attn1, attn2
    
class Decoder(nn.Module):

    def __init__(self, dm, nhead, dff, layers=6, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.stack = nn.ModuleList([DecoderLayer(dm, nhead, dff, bias, dropout, eps) \
            for l in range(layers)])

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape src: (batch_size, seq_len, d_model) tgt: (batch_size, tgt_len, d_model)
        x = tgt
        # pass src & tgt through stack of decoders (out of layer is in for next)
        for decoder in self.stack:
            x, attn1, attn2 = decoder(src, x, src_mask=src_mask, tgt_mask=tgt_mask)
        out = x
        return out, attn1, attn2

if __name__ == "__main__":
    pass
    


