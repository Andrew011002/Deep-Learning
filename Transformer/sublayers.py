import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dk, dropout=0.1) -> None:
        super().__init__()
        self.norm = 1 / torch.sqrt(torch.Tensor([dk])).item()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # inputs are projected shape = q: (batch_size, q_len, d_k) k & v : (batch_size, seq_len, d_k)
        similarities = torch.matmul(q, k.transpose(-2, -1)) * self.norm

        # apply mask (if required)
        if mask is not None:
            similarities = similarities.masked_fill(mask == 0, -1e9)

        # compute attention weights
        attention = self.softmax(similarities)
        attention = self.dropout(attention)

        # compute context
        context = torch.matmul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, dm, nhead, bias=False, attn_dropout=0.1, dropout=0.1) -> None:
        super().__init__()

        if dm % nhead != 0:
            raise ValueError("Embedding dimensions (dm) must be divisble by number of heads (nheads.")
        
        self.dm = dm
        self.dk = dm // nhead
        self.nhead = nhead
        self.wq = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wk = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wv = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wo = nn.Linear(dm, dm)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(self.dk, dropout=attn_dropout)

    def forward(self, q, k, v, mask=None):
        # inshape: q = (batch_size, q_len, dm) k & v = (batch_size, seq_len, dm)
        batch_size = q.size(0)

        # linear projections into heads shape: q = (batch_size, nheads, q_len, dk) k & v = (batch_size, nheads, seq_len, dk)
        q = self.wq(q).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)

        # att scores & weights shape: attention = (batch_size, nhead, q_len, seq_len) values = (batch_size, nhead, q_len, dk)
        context, attention = self.scaled_dot_prod_attn(q, k, v, mask=mask)

        # concat shape: attention (batch_size, nheads, q_len, seq_len) context = (batch_size, q_len, dm)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dm)

        # project and dropout shape: context = (batch_size, q_len, dm)
        context = self.wo(context)
        context = self.dropout(context)
        return context, attention


class Norm(nn.Module):

    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        # input shape: (batch_size, seq_len, d_model)

        # calculate mean & standard deviation
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        std = torch.sqrt(var + self.epsilon)

        # normalize
        norm = (x - mean) / std * self.scale + self.bias
        return norm


if __name__ == "__main__":
    q = torch.rand((64, 10, 512))
    k = torch.rand((64, 10, 512))
    v = k

    # MULTI-HEAD CALCULATION
    multihead = MultiHeadAttention(512, 8)
    values, attention = multihead(q, k, v, mask=None)
    print(attention.size(), values.size())

    # NORM LAYER
    norm = Norm(512)
    normed = norm(values)
    print(normed.size())



    
    
    




    

    

