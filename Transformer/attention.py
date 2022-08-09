import torch
import torch.nn as nn
import numpy as np


class AttentionHead(nn.Module):

    def __init__(self, dk, dropout=0.1) -> None:
        super().__init__()
        # weights for q,k, & v shape : (dk, dk)
        self.wq = nn.Linear(dk, dk)
        self.wk = nn.Linear(dk, dk)
        self.wv = nn.Linear(dk, dk)
        self.dk = torch.Tensor([dk])
        self.dropout = dropout
        

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # inshapes = q : (batch_size, 1, dk) k, v : (batch_size, seq_len, dk)
        batch_size = q.size(0)

        # initial linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # scaled dot product
        norm = 1 / torch.sqrt(self.dk)
        s = torch.matmul(q, k.view((batch_size, int(self.dk.item()), -1))) # (batch_size, 1, seq_len)    
        s /= norm
        
        # normalization
        a = torch.exp(s) / torch.sum(torch.exp(s), dim=2, keepdim=True) # (batch_size, 1, seq_len)
        a = self.dropout(a)

        # matmul for for attention values
        attention = torch.matmul(a, v) # (batch_size, 1, dk)
        return attention


if __name__ == "__main__":
    q = torch.rand((1, 1, 5))
    k = torch.rand((1, 6, 5))
    v = k

    head = AttentionHead(5, 5)
    attention = head(q, k, v)
    print(attention.size())