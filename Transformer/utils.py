import torch
import torch.nn as nn

def create_subsequent_mask(seq):
    # inshape: (batch_size, seq_len)
    batch_size, seq_len = seq.size(0), seq.size(1)

    # create a diagnal of 1's (left) 0's (right)
    mask = torch.triu(torch.ones(seq_len, seq_len) == 1)
    return mask.int().transpose(0, 1)

def create_padded_mask(seq, pad_val):
    # inshape: (batch_size, seq_len)
    return seq != pad_val


if __name__ == "__main__":
    pass