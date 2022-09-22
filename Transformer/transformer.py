import torch
import torch.nn as nn
from embedding import Embeddings
from pos_encoder import PositionalEncoder
from layers import Encoder, Decoder
from utils import generate_mask



class Transformer(nn.Module):
    
    def __init__(self, n_tokens, maxlen, pad_idx=0, dm=512, nhead=8, layers=6, 
                    dff=2048, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.embeddings = Embeddings(n_tokens, dm, pad_idx)
        self.pos_encoder = PositionalEncoder(dm, maxlen, dropout)
        self.encoders = nn.ModuleList([Encoder(dm, nhead, dff, bias, dropout, eps) \
                                        for l in range(layers)])
        self.decoders = nn.ModuleList([Decoder(dm, nhead, dff, bias, dropout, eps) \
                                        for l in range(layers)])
        self.wu = self.embeddings.linear()
        self.softmax = nn.Softmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, softmax=False):
        # inshape: (batch_size, seq_len)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(src)
        x = self.pos_encoder(x)

        # encode embeddings shape: (batch_size, seq_len, dm)
        for encoder in self.encoders:
            x, e_attn = encoder(x, src_mask=src_mask)
        e_out = x

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(tgt)        
        x = self.pos_encoder(x)

        # decode embeddings shape: (batch_size, seq_len, dm)
        for decoder in self.decoders:
            x, d_attn1, d_attn2 = decoder(e_out, x, src_mask=src_mask, tgt_mask=tgt_mask)
        d_out = x

        # linear transform & softmax shape: (batch_size, tgt_len, n_tokens)
        out = torch.matmul(d_out, self.wu.T)

        # apply sm (inference) leave alone (train)
        return self.softmax(out) if softmax else out

if __name__ == "__main__":
    pass

    
        

    
    


    