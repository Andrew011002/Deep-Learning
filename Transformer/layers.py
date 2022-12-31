import torch.nn as nn
from embedding import Embeddings
from pos_encoder import PositionalEncoder
from sublayers import MultiHeadAttention, Norm, FeedForwardNetwork


class EncoderLayer(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(dm, nhead, bias=bias, dropout=dropout)
        self.norm1 = Norm(dm, eps=eps)
        self.norm2 = Norm(dm, eps=eps)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout=dropout)

    def forward(self, src, src_mask=None):
        # inshape: src - (batch_size, seq_len, dm)

        # get context, add & norm (residual connections) shape: x - (batch_size, seq_len, dm)
        x = src
        x_out, attn = self.multihead(x, x, x, mask=src_mask)
        x = self.norm1(x + x_out)

        # calc linear transforms, add & norm (residual connections) shape: out - (batch_size, seq_len, dm)
        x_out = self.feedforward(x)
        out = self.norm2(x + x_out)
        return out, attn

class Encoder(nn.Module):

    def __init__(self, vocab_size, maxlen, pad_id, dm, nhead, dff,
                    layers=6, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.stack = nn.ModuleList([EncoderLayer(dm, nhead, dff, bias=bias, dropout=dropout, eps=eps) \
            for l in range(layers)])
        self.embeddings = Embeddings(vocab_size, dm, pad_id=pad_id)
        self.pos_encodings = PositionalEncoder(dm, maxlen, dropout=dropout)

    def forward(self, src, src_mask=None):
        # inshape: src - (batch_size, seq_len, d_model)

        # embeddings + positional encodings shape: x - (batch_size, seq_len, dm)
        x = self.embeddings(src)
        x = self.pos_encodings(x)
        # pass src through stack of encoders (out of layer is in for next)
        for encoder in self.stack:
            x, attn = encoder(x, src_mask=src_mask)
        out = x
        return out, attn

class DecoderLayer(nn.Module):

    def __init__(self, dm, nhead, dff, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.maskmultihead = MultiHeadAttention(dm, nhead, bias=bias, dropout=dropout)
        self.multihead = MultiHeadAttention(dm, nhead, bias=bias, dropout=dropout)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout=dropout)
        self.norm1 = Norm(dm, eps=eps)
        self.norm2 = Norm(dm, eps=eps)
        self.norm3 = Norm(dm, eps=eps)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape: src - (batch_size seq_len, dm) tgt - (batch_size, tgt_len, dm)

        # calc masked context, add & norm (residual connections) shape: x - (batch_size, tgt_len, dm)
        x = tgt
        x_out, attn1 = self.maskmultihead(x, x, x, mask=tgt_mask)
        x = self.norm1(x + x_out)

        # calc context, add & norm (residual connections) shape: x - (batch_size, tgt_len, dm)
        x_out, attn2 = self.multihead(x, src, src, mask=src_mask)
        x = self.norm2(x + x_out)

        # calc linear transforms, add & norm (residual connections) shape: out - (batch_size, tgt_len, dm)
        x_out = self.feedforward(x)
        out = self.norm3(x + x_out)
        return out, attn1, attn2
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, maxlen, pad_id, dm, nhead, dff, 
                    layers=6, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.stack = nn.ModuleList([DecoderLayer(dm, nhead, dff, bias=bias, dropout=dropout, eps=eps) \
            for l in range(layers)])
        self.embeddings = Embeddings(vocab_size, dm, pad_id=pad_id)
        self.pos_encodings = PositionalEncoder(dm, maxlen, dropout=dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape: src - (batch_size, seq_len, dm) tgt - (batch_size, tgt_len, dm)

        # embeddings + positional encodings shape: x - (batch_size, seq_len, dm)
        x = self.embeddings(tgt)
        x = self.pos_encodings(x)
        
        # pass src & tgt through stack of decoders (out of layer is in for next)
        for decoder in self.stack:
            x, attn1, attn2 = decoder(src, x, src_mask=src_mask, tgt_mask=tgt_mask)
        out = x
        return out, attn1, attn2

if __name__ == "__main__":
    pass
    


