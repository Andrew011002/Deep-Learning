import torch
import torch.nn as nn
from embedding import Embeddings
from pos_encoder import PositionalEncoder
from layers import Encoder, Decoder



class Transformer(nn.Module):
    
    def __init__(self, n_tokens, max_len, pad_idx=0, dm=512, nhead=8, layers=6, 
                    dff=2048, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.embeddings = Embeddings(n_tokens, dm, pad_idx)
        self.pos_encoder = PositionalEncoder(dm, max_len, dropout)
        self.encoder = Encoder(dm, nhead, dff, bias, dropout, eps)
        self.decoder = Decoder(dm, nhead, dff, bias, dropout, eps)
        self.linear = self.embeddings.linear()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, ouputs):
        # inshape: (batch_size, seq_len)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(inputs)
        x = self.pos_encoder(x)

        # encode embeddings shape: (batch_size, seq_len, dm)
        e_out, e_attn = self.encoder(x, src_mask=None)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(outputs)        
        x = self.pos_encoder(x)

        # decode embeddings shape: (batch_size, seq_len, dm)
        d_out, d_attn1, d_attn2 = self.decoder(e_out, x, src_mask=None, tgt_mask=None)

        # linear transform & softmax shape: (batch_size, seq_len, n_tokens)
        out = torch.matmul(d_out, self.linear.T)
        return self.softmax(out)

def create_subsequent_mask(seq):
    seq_len = seq.size(1)
    # create a diagnal of 1's (left) 0's (right)
    mask = torch.triu(torch.ones(seq_len, seq_len) == 1)
    return mask.int().transpose(0, 1)

def create_padded_mask(seq, pad_val):
    return seq != pad_val

def create_mask(src, tgt, pad_val):
    src_mask = create_padded_mask(src, pad_val)
    tgt_mask = create_padded_mask(tgt, pad_val) & create_subsequent_mask(tgt)
    return src_mask, tgt_mask


if __name__ == "__main__":
    vocab_size = 10000
    seq_len = 25
    batch_size = 32
    pad_idx = 0

    inputs = torch.randint(0, vocab_size, (32, seq_len))
    outputs = inputs[:, 1:]

    transformer = Transformer(vocab_size, seq_len)
    out = transformer(inputs, outputs)

    print(out.size())