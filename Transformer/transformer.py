import torch
import torch.nn as nn
from embedding import Embeddings
from pos_encoder import PositionalEncoder
from layers import Encoder, Decoder

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, maxlen, pad_id, dm=512, nhead=8, layers=6, 
                    dff=2048, bias=False, dropout=0.1, eps=1e-5) -> None:
        super().__init__()
        self.embeddings = Embeddings(vocab_size, dm, pad_id)
        self.pos_encoder = PositionalEncoder(dm, maxlen, dropout)
        self.encoder = Encoder(dm, nhead, dff, layers, bias, dropout, eps)          
        self.decoder = Decoder(dm, nhead, dff, layers, bias, dropout, eps)
        self.wu = self.embeddings.unembedding()
        self.softmax = nn.Softmax(dim=-1)
        self.pad_id = pad_id
        self.xavier_init()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # inshape: (batch_size, seq_len)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(src)
        x = self.pos_encoder(x)

        # encode embeddings shape: (batch_size, seq_len, dm)
        e_out, attn = self.encoder(x, src_mask=src_mask)

        # embeddings + positional encodings shape: (batch_size, seq_len, dm)
        x = self.embeddings(tgt)        
        x = self.pos_encoder(x)

        # decode embeddings shape: (batch_size, seq_len, dm)
        d_out, attn1, attn2 = self.decoder(e_out, x, src_mask=src_mask, tgt_mask=tgt_mask)

        # umbedding wieth embedding weight matrix
        out = torch.matmul(d_out, self.wu.T)
        return out
        
    def xavier_init(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

if __name__ == "__main__":
    pass

    
        

    
    


    