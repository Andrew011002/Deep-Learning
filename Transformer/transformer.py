import torch
import torch.nn as nn
from embedding import Embeddings
from pos_encoder import PositionalEncoder
from layers import Encoder, Decoder
from utils import generate_mask



class Transformer(nn.Module):
    
    def __init__(self, n_tokens, max_len, pad_idx=0, dm=512, nhead=8, layers=6, 
                    dff=2048, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.embeddings = Embeddings(n_tokens, dm, pad_idx)
        self.pos_encoder = PositionalEncoder(dm, max_len, dropout)
        self.encoders = nn.ModuleList([Encoder(dm, nhead, dff, bias, dropout, eps) \
                                        for _ in range(layers)])
        self.decoders = nn.ModuleList([Decoder(dm, nhead, dff, bias, dropout, eps) \
                                        for _ in range(layers)])
        self.wu = self.embeddings.linear()
        self.softmax = nn.Softmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, inputs, outputs, src_mask=None, tgt_mask=None, softmax=False):
        # inshape: (batch_size, seq_len)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(inputs)
        x = self.pos_encoder(x)

        # encode embeddings shape: (batch_size, seq_len, dm)
        for encoder in self.encoders:
            x, e_attn = encoder(x, src_mask=src_mask)
        e_out = x

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(outputs)        
        x = self.pos_encoder(x)

        # decode embeddings shape: (batch_size, seq_len, dm)
        for decoder in self.decoders:
            x, d_attn1, d_attn2 = decoder(e_out, x, src_mask=src_mask, tgt_mask=tgt_mask)
        d_out = x

        # linear transform & softmax shape: (batch_size, tgt_len, n_tokens)
        out = torch.matmul(d_out, self.wu.T)
        # apply softmax (for inference)
        if softmax:
            return self.softmax(out)
        return out

if __name__ == "__main__":
    vocab_size = 1000
    seq_len = 25
    batch_size = 16
    pad_idx = 0
    dm = 512

    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    outputs = inputs[:, :-1]

    transformer = Transformer(vocab_size, seq_len)
    out = transformer(inputs, outputs)
    print(out.size())

    
        

    
    


    