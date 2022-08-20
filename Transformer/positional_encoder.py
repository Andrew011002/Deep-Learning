import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_len, p_drop=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p_drop)

        # pos shape: (max seq_len, 1) dim: (d_model, )
        pos = torch.arange(max_len).float().view(-1, 1)
        dim = torch.arange(d_model).float()

        # apply pos / (10000^2*i / d_model) -> use sin for even indices & cosine for odd indices
        values = pos / torch.pow(1e4, 2 * torch.div(dim, 2, rounding_mode="floor") / d_model)
        encodings = torch.where(dim.long() % 2 == 0, torch.sin(values), torch.cos(values))

        # reshape: (batch_size, max seq_len, d_model) -> register encodings w/o grad
        encodings = encodings.unsqueeze(0)
        self.register_buffer("pos_encodings", encodings)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, seq_len, d_model)
        seq_len = embeddings.size(1)
        
        # sum up encodings up to seq_len shape: (batch_size, seq_len, d_model)
        embeddings = embeddings + self.pos_encodings[:, :seq_len]
        return self.dropout(embeddings)

if __name__ == '__main__':
    batch_size = 64
    seq_len = 15
    max_len = 20
    d_model = 512
    
    embeddings = torch.rand(batch_size, seq_len, d_model)
    pos_encoder = PositionalEncoder(d_model, max_len, p_drop=0.1)

    new_embeddings = pos_encoder(embeddings)
    print(new_embeddings.size())

    
