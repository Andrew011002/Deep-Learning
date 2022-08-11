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

class AttentionHead(nn.Module):

    def __init__(self, dm, nhead=1, bias=False, dropout=0.1) -> None:
        super().__init__()
        
        if dm % nhead != 0:
            raise ValueError("The embeddings dimenions (dm) must be divisble by number of heads (nhead).")
        
        self.dm = dm
        self.dk = dm // nhead
        self.nhead = nhead
        self.wq = nn.Linear(dm, dm, bias=bias)
        self.wk = nn.Linear(dm, dm, bias=bias)
        self.wv = nn.Linear(dm, dm, bias=bias)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(self.dk, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        # inshpe : q = (batch_size, q_len, dm) k & v = (batch_size, seq_len, dm)
        batch_size = q.size(0)

        # project q, k, & v reshape: q = (batch_size, nhead, q_len, dk) k & v = (batch_size, nhead, seq_len, dk)
        q = self.wq(q).view(batch_size, -1, self.nhead, self.dk).permute(0, 2, 1, 3)
        k = self.wk(k).view(batch_size, -1, self.nhead, self.dk).permute(0, 2, 1, 3)
        v = self.wv(v).view(batch_size, -1, self.nhead, self.dk).permute(0, 2, 1, 3)

        # scaled dot prod attn shape: attention = (batch_size, nhead, q_len, seq_len) values = (batch_size, nhead, q_len, dk)
        values, attention = self.scaled_dot_prod_attn(q, k, v, mask=mask)

        # reshape: values = (batch_size, q_len, dm)
        values = values.contiguous().view(batch_size, -1, self.dm)
        return values, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, dm, nhead, bias=False, attn_dropout=0.1, dropout=0.1) -> None:
        super().__init__()

        if nhead <= 1:
            raise ValueError("Number of heads (nheads) must greater than 1. Use AttentionHead for single head use only.")

        self.wo = nn.Linear(dm, dm)
        self.heads = AttentionHead(dm, nhead=nhead, bias=bias, dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # inshpe : q = (batch_size, q_len, dm) k & v = (batch_size, seq_len, dm)

        # calculate values & attention
        values, attention = self.heads(q, k, v, mask=mask)

        # projection values shape: values = (batch_size, q_len, dm) attention = (batch_size, nhead, q_len, seq_len)
        values = self.wo(values)
        values = self.dropout(values)
        return values, attention

if __name__ == "__main__":
    q = torch.rand((32, 1, 512))
    k = torch.rand((32, 10, 512))
    v = k

    # SINGLE HEAD CALCULATION
    head = AttentionHead(512)
    values, attention = head(q, k, v)
    print(attention.size())
    print(values.size())

    q = torch.rand((64, 5, 512))
    k = torch.rand((64, 10, 512))
    v = k

    # MULTI-HEAD CALCULATION
    multihead = MultiHeadAttention(512, 8)
    values, attention = multihead(q, k, v)
    print(attention.size())
    print(values.size())
