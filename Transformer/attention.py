import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dk, dropout=0.1) -> None:
        super().__init__()
        self.norm = 1 / torch.sqrt(torch.Tensor([dk]))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # inputs are projected shape = q: (batch_size, q_len, d_k) k & v : (batch_size, seq_len, d_k)
        similarities = torch.matmul(q, k.transpose(-2, -1)) * self.norm
        attention = self.softmax(similarities)
        attention = self.dropout(attention)
        values = torch.matmul(attention, v)
        return values, attention

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
        values, attention = self.scaled_dot_prod_attn(q, k, v, mask=mask)

        # concat shape: attention (batch_size, nheads, q_len, seq_len) values = (batch_size, q_len, dm)
        values = values.transpose(1, 2).contiguous().view(batch_size, -1, self.dm)

        # project and dropout shape: values = (batch_size, q_len, dm)
        values = self.wo(values)
        values = self.dropout(values)
        return values, attention


if __name__ == "__main__":
    q = torch.rand((64, 5, 512))
    k = torch.rand((64, 10, 512))
    v = k

    # MULTI-HEAD CALCULATION
    multihead = MultiHeadAttention(512, 8)
    values, attention = multihead(q, k, v)
    print(attention.size(), values.size())
    

