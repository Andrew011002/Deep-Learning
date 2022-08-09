import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, seq_length, dropout=0.3) -> None:
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # create numerator & denominator for sinusoidal waves
        pos = torch.arange(seq_length).float().view(-1, 1)
        dim = torch.arange(d_model).float()

        # apply pos / (10000^2*i / d_model) -> use sin for even indices & cosine for odd indices
        values = pos / torch.pow(1e4, 2 * torch.div(dim, 2, rounding_mode="floor") / d_model)
        encodings = torch.where(dim.long() % 2 == 0, torch.sin(values), torch.cos(values))

        # reshape: (seq_len, batch_size, d_model) -> register encodings w/o buffer
        encodings = encodings.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encodings", encodings)

    def forward(self, embeddings):
        # sum embeddings with positional encodings for entire batch
        return self.dropout(embeddings + self.pos_encodings[:embeddings.size(0), :])
    
if __name__ == '__main__':
    positional_encoder = PositionalEncoder(8, 4)
    
