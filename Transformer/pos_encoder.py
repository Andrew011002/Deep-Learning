import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):

    def __init__(self, dm, maxlen, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # shape: pos - (max_len, 1) dim - (dm, )
        pos = torch.arange(maxlen).float().view(-1, 1)
        dim = torch.arange(dm).float()

        # apply pos / (10000^2*i / dm) -> use sin for even indices & cosine for odd indices
        values = pos / torch.pow(1e4, 2 * torch.div(dim, 2, rounding_mode="floor") / dm)
        encodings = torch.where(dim.long() % 2 == 0, torch.sin(values), torch.cos(values))

        # reshape: encodings - (batch_size, max seq_len, dm)
        encodings = encodings.unsqueeze(0)
        # register encodings w/o grad
        self.register_buffer("pos_encodings", encodings)

    def forward(self, embeddings):
        # inshape: embeddings - (batch_size, seq_len, dm)

        # sum embeddings w/ respective positonal encodings shape: embeddings - (batch_size, seq_len, dm)
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_encodings[:, :seq_len]
        # drop neurons
        return self.dropout(embeddings)

if __name__ == '__main__':
    pass


    
