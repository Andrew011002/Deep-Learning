import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dk) -> None:
        super().__init__()
        self.norm = 1 / np.sqrt(dk) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # inputs are projected shape: q (batch_size, q_len, dk) k & v (batch_size, k_len, dk)

        # compute dot prod w/ q & k then normalize shape: similarities (batch_size, q_len, k_len)
        similarities = torch.matmul(q, k.transpose(-2, -1)) * self.norm

        # apply mask (if required)
        if mask is not None:
            mask = mask.unsqueeze(1) # for multi-head attention
            similarities = similarities.masked_fill(mask == 0, -1e9)

        # compute attention weights
        attention = self.softmax(similarities)

        # compute context given v
        context = torch.matmul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, dm, nhead, bias=False, dropout=0.1) -> None:
        super().__init__()

        if dm % nhead != 0:
            raise ValueError("Embedding dimensions (dm) must be divisble by number of heads (nheads)")
        
        self.dm = dm
        self.dk = dm // nhead
        self.nhead = nhead
        self.wq = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wk = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wv = nn.Linear(dm, self.dk * nhead, bias=bias)
        self.wo = nn.Linear(dm, dm)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(self.dk)

    def forward(self, q, k, v, mask=None):
        # inshape: q = (batch_size, q_len, dm) k & v = (batch_size, k_len, dm)
        batch_size = q.size(0)

        # linear projections into heads shape: q (batch_size, nheads, q_len, dk) k & v (batch_size, nheads, k_len, dk)
        q = self.wq(q).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.nhead, self.dk).transpose(1, 2)

        # attn scores & weights shape: attention (batch_size, nhead, q_len, k_len) values (batch_size, nhead, q_len, dk)
        context, attention = self.scaled_dot_prod_attn(q, k, v, mask=mask)

        # concat shape: attention (batch_size, nheads, q_len, k_len) context (batch_size, q_len, dm)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dm)

        # project & drop neurons shape: context (batch_size, q_len, dm)
        context = self.wo(context)
        context = self.dropout(context)
        return context, attention

class Norm(nn.Module):

    def __init__(self, dm, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dm))
        self.beta = nn.Parameter(torch.zeros(dm))
        self.eps = eps

    def forward(self, x):
        # input shape: (batch_size, seq_len, dm)

        # calculate mean & standard deviation (along dm)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        # normalize
        norm = (x - mean) / std
        return norm * self.gamma + self.beta

class FeedForwardNetwork(nn.Module):

    def __init__(self, dm, dff, dropout=0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(dm, dff)
        self.w2 = nn.Linear(dff, dm)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inshape (batch_size, seq_len, dm)
        
        # first linear transform with ReLU shape: (batch_size, seq_len, dff)
        x = self.relu(self.w1(x))

        # second linear transform shape: (batch_size, seq_len, dm)
        x = self.w2(x)
        # drop neurons
        out = self.dropout(x)
        return out


if __name__ == "__main__":
    pass