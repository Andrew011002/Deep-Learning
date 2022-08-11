import torch
import torch.nn as nn

class AttentionHead(nn.Module):

    def __init__(self, dk,bias=False, dropout=0.1) -> None:
        super().__init__()
        # weights for q,k, & v shape : (dk, dk)
        self.wq = nn.Linear(dk, dk, bias=bias)
        self.wk = nn.Linear(dk, dk, bias=bias)
        self.wv = nn.Linear(dk, dk, bias=bias)
        self.dk = torch.Tensor([dk])
        self.dropout = nn.Dropout(dropout) # for dropping scores
        

    def forward(self, q, k, v, mask=None):
        # inshapes (single head) = q : (batch_size, 1, dk) k, v : (batch_size, seq_len, dk)
        # inshapes (multi head) = q : (batch_size, 1, nhead, dk) k, v : (batch_size, seq_len, nhead, dk)

        # initial linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # scaled dot product
        norm = 1 / torch.sqrt(self.dk)
        s = torch.matmul(q, k.transpose(-2, -1)) * norm # (batch_size, 1, seq_len)    

        # normalization
        attention = torch.exp(s) / torch.sum(torch.exp(s), dim=-1, keepdim=True) # (batch_size, 1, seq_len)
        attention = self.dropout(attention)

        # matmul for for attention values
        output = torch.matmul(attention, v) # (batch_size, 1, dk)
        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, nhead, dk, bias=False, dropout=0.1) -> None:
        super().__init__()
        # heads must be evenly divisible by dk
        if dk % nhead != 0:
            raise ValueError("nhead must be a modulo of dk")

        # create single head derived from even splits, weight out shape: (dk, dk)
        self.dk = dk
        self.dh = dk // nhead
        self.nhead = nhead
        self.head = AttentionHead(self.dh, bias, dropout)
        self.wo = nn.Linear(dk, dk, bias=bias)

    def forward(self, q, k, v, mask=None):
        # inshapes (single head) = q : (batch_size, 1, dk) k, v : (batch_size, seq_len, dk)
        batch_size = q.size(0)

        # split into nheads shape = q : (batch_size, 1, nhead, dh) k & v : (batch_size, seq_len, nhead, dh)
        # -> attention head reshape = q : (batch_size, nhead, 1, dh) k & v : (batch_size, nhead, seq_len, dh)
        q = q.reshape(batch_size, -1, self.nhead, self.dh).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.nhead, self.dh).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.nhead, self.dh).permute(0, 2, 1, 3)

        # calc attention shape : (batch_size, nhead, 1, dh)
        attention = self.head(q, k, v, mask=mask)
        
        # concat
        concat = attention.reshape(batch_size, -1, self.dk)

        # project
        projection = self.wo(concat)
        return projection


if __name__ == "__main__":
    q = torch.rand((32, 1, 512))
    k = torch.rand((32, 10, 512))
    v = k

    head = AttentionHead(512)
    attention = head(q, k, v)
    print(attention.size())

    q = torch.rand((64, 1, 512))
    k = torch.rand((64, 10, 512))
    v = k

    multihead = MultiHeadAttention(8, 512)
    attention = multihead(q, k, v)
    print(attention.size())
