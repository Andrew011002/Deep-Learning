import torch
import torch.nn as nn


def create_subsequent_mask(seq):
    seq_len = seq.size(1)
    # create a diagnal of 1's (left) 0's (right)
    mask = torch.triu(torch.ones(seq_len, seq_len) == 1)
    return mask.int().transpose(0, 1)

def create_padded_mask(seq, pad_val):
    return seq != pad_val

def create_mask(src, tgt, pad_val):
    src_mask = create_padded_mask(src, pad_val)
    tgt_mask = create_padded_mask(tgt, pad_val) & create_subsequent_mask(tgt)
    return src_mask, tgt_mask