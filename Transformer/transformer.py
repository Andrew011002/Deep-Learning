import torch
import torch.nn as nn
from layers import Encoder, Decoder

class Transformer(nn.Module):
    
    def __init__(self, vocab_enc, vocab_dec, maxlen, pad_id, dm=512, nhead=8, layers=6, 
                    dff=2048, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_enc, maxlen, pad_id, dm, nhead, dff, \
            layers=layers, bias=bias, dropout=dropout, eps=eps)          
        self.decoder = Decoder(vocab_dec, maxlen, pad_id, dm, nhead, dff, \
            layers=layers, bias=bias, dropout=dropout, eps=eps)
        self.linear = self.decoder.embeddings.unembedding()
        self.maxlen = maxlen
        self.pad_id = pad_id
        self.xavier_init()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape: src - (batch_size, src_len) tgt - (batch_size, tgt_len)

        # encode embeddings shape: e_out - (batch_size, src_len, dm)
        e_out, attn = self.encoder(src, src_mask=src_mask)

        # decode embeddings shape: d_out - (batch_size, tgt_len, dm)
        d_out, attn, attn = self.decoder(e_out, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # linear project out of decoder: out (batch_size, tgt_len, vocab_size)
        out = torch.matmul(d_out, self.linear.T)
        return out
        
    def xavier_init(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

if __name__ == "__main__":
    pass

    
        

    
    


    