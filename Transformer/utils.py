import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_masks(source, targets, pad_idx):
    # inshape source: (batch_size, inputs_len) targets: (batch_size, outputs_len) pad_idx: (,)
    tgt_len = targets.size(1)

    # create padded mask for src & tgt 
    src_mask = (source != pad_idx).unsqueeze(-2)
    tgt_mask = (targets != pad_idx).unsqueeze(-2)

    # create subsequent mask for tgt (no peak) shape tgt_nopeak_mask: (1, tgt_len, tgt_len)
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    
    # combine tgt_pad_mask & tgt_nopeak_mask to hide pad and prevent subsequent attention
    tgt_mask = tgt_mask & tgt_nopeak_mask
    # shape src_mask: (batch_size, 1, seq_len) tgt_mask: (batch_size, tgt_len, tgt_len)
    return src_mask, tgt_mask

def generate_nopeak_pad_mask(tgt, pad_idx):
    # inshape tgt: (batch_size, tgt_len)
    tgt_len = tgt.size(1)
    # padded mask (True where no pad False otherwise)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    # create subsequent mask
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    # combine padded & subsequent mask
    tgt_mask = tgt_mask & tgt_nopeak_mask
    return tgt_mask

def pad_tokens(tokens, pad_idx, end=True):
    # inshape (n, arbitrary)
    padded = []
    # find max len in tokens
    maxlen = len(max(tokens, key=len))
    # pad all seqs in tokens
    for seq in tokens:
        seq = np.array(seq)
        seq = pad_sequence(seq, maxlen, pad_idx, end=end)
        padded.append(seq)
    # return as array of ints
    return np.array(padded, dtype=np.float64)

def pad_sequence(sequence, maxlen, pad_idx, end=True):
    seq_len = len(sequence)
    # pad the sequence if it's smaller than maxlen
    if seq_len < maxlen:
        # create pad
        pad = np.zeros((maxlen - seq_len,)) + pad_idx
        # add pad to start or end
        sequence = np.append(sequence, pad) if end else np.append(pad, sequence)
    return sequence

def pad_outputs(tokens_1, tokens_2, pad_idx, end=True):
    # find max len between the pair of tokens
    maxlen = max(len(max(tokens_1, key=len)), len(max(tokens_2, key=len)))
    padded_1, padded_2 = [], []
    # pad all seqs in both tokens with same maxlen
    for seq_1, seq_2 in zip(tokens_1, tokens_2):
        seq_1, seq_2 = np.array(seq_1), np.array(seq_2)
        seq_1 = pad_sequence(seq_1, maxlen, pad_idx, end)
        seq_2 = pad_sequence(seq_2, maxlen, pad_idx, end)
        padded_1.append(seq_1), padded_2.append(seq_2)
        # return as array of ints for both 
    return np.array(padded_1, dtype=np.int64), np.array(padded_2, dtype=np.int64)

def truncate_tokens(tokens, maxlen, delete=False):
    truncated = []
    # delete seqs that exceed maxlen
    if delete:
        for seq in tokens:
            # only add seq if in limits of maxlen
            if len(seq) <= maxlen:
                truncated.append(seq)
    # truncate seqs that exceed maxlen
    else:
        for seq in tokens:
            seq = np.array(seq)
            # truncate seq larger than maxlen
            if len(seq) > maxlen:
                seq = seq[:maxlen]
            truncated.append(seq)
    # return as array of objects
    return np.array(truncated, dtype=object)

def truncate_outputs(tokens_1, tokens_2, maxlen, delete=False):
    truncated_1, truncated_2 = [], []
    # delete seqs that exceed maxlen
    if delete:
        for seq_1, seq_2 in zip(tokens_1, tokens_2):
            # only add seq pair if in limits of maxlen
            if len(seq_1) <= maxlen >= len(seq_2):
                truncated_1.append(seq_1), truncated_2.append(seq_2)
        return np.array(truncated_1, dtype=object), np.array(truncated_2, dtype=object)
    # truncate seqs that exceed maxlen and return as array of objects for both
    return truncate_tokens(tokens_1, maxlen, delete=False), truncate_tokens(tokens_2, maxlen, delete=False)

def ascending_data(vocab_size, n, maxlen, sos, eos, pad_idx):
    inputs, outputs = [], []
    base = np.array([sos, eos])
    token_set = set(range(n))
    for i in range(n):
        seq_len = np.random.randint(1, maxlen + 1)
        token = np.random.randint(1, vocab_size) - maxlen
        token *= -1 if token < 0 else 1
        seq = np.arange(token, token + seq_len)
        seq = np.insert(base, 1, seq)
        src, tgt = seq, seq[::-1]
        inputs.append(src), outputs.append(tgt)

    inputs = pad_tokens(inputs, pad_idx=pad_idx, end=True)
    outputs = pad_tokens(outputs, pad_idx=pad_idx, end=True)
    return inputs, outputs
        
def create_dataloader(inputs, labels, batch_size=32, drop_last=True, shuffle=False, **dataloader_kwargs):
    # create tensors
    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
    tensorset = TensorDataset(inputs, labels)
    # create dataloader with specified args
    dataloader = DataLoader(tensorset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, **dataloader_kwargs)
    return dataloader

if __name__ == "__main__":
    pass
    