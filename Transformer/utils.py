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
    tgt_len = tgt.size(1)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    tgt_mask = tgt_mask & tgt_nopeak_mask
    return tgt_mask

def pad_tokens(tokens, pad_idx, end=True):
    padded = []
    maxlen = len(max(tokens, key=len))
    for seq in tokens:
        seq = np.array(seq)
        seq = pad_sequence(seq, maxlen, pad_idx, end=end)
        padded.append(seq)
    return np.array(padded, dtype=np.float64)

def pad_sequence(sequence, maxlen, pad_idx, end=True):
    seq_len = len(sequence)
    if seq_len < maxlen:
        pad = np.zeros((maxlen - seq_len,)) + pad_idx
        sequence = np.append(sequence, pad) if end else np.append(pad, sequence)
    return sequence

def truncate_tokens(tokens, maxlen):
    # modify tokens inplace
    for i, seq in enumerate(tokens):
        # truncate seq larger than maxlen
        if len(seq) > maxlen:
            seq = seq[:maxlen]
            tokens[i] = seq
        
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
    
    

    
    


    
    
    
    





    
    
    

    







    
    



    