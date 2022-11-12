import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from tokenizer import WordPieceTokenizer

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

        
class Dataset(IterableDataset):
    
    def __init__(self, inputs, labels, vocab_size, tokenizer_kwargs=None):
        self.size = len(inputs)
        self.data = inputs, labels
        self. tokenizer = self.__init_tokenizer(inputs, labels, vocab_size)
        self.inputs = self.__tokenize(inputs)
        self.labels = self.__tokenize(labels)

    def __init_tokenizer(self, inputs, labels, vocab_size):
        tokenizer = WordPieceTokenizer()
        corpus = list(inputs) + list(labels)
        tokenizer.train(vocab_size, corpus)
        return tokenizer

    def __tokenize(self, data):
        return self.tokenizer.encode(data, model=True), self.tokenizer.encode(data, model=False)

    def get_inputs(self, model=False):
        if model:
            return self.inputs[0]
        return self.inputs[1]

    def get_labels(self, model=False):
        if model:
            return self.labels[0]
        return self.labels[1]

    def get_tokenizer(self):
        return self.tokenizer

    
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

    