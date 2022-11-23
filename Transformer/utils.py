import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from datasets import load_dataset

def generate_masks(source, targets, pad_id):
    # inshape source: (batch_size, inputs_len) targets: (batch_size, outputs_len) pad_id: (,)
    tgt_len = targets.size(1)

    # create padded mask for src & tgt 
    src_mask = (source != pad_id).unsqueeze(-2)
    tgt_mask = (targets != pad_id).unsqueeze(-2)

    # create subsequent mask for tgt (no peak) shape tgt_nopeak_mask: (1, tgt_len, tgt_len)
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    
    # combine tgt_pad_mask & tgt_nopeak_mask to hide pad and prevent subsequent attention
    tgt_mask = tgt_mask & tgt_nopeak_mask
    # shape src_mask: (batch_size, 1, seq_len) tgt_mask: (batch_size, tgt_len, tgt_len)
    return src_mask, tgt_mask

def generate_nopeak_pad_mask(tgt, pad_id):
    # inshape tgt: (batch_size, tgt_len)
    tgt_len = tgt.size(1)
    # padded mask (True where no pad False otherwise)
    tgt_mask = (tgt != pad_id).unsqueeze(-2)
    # create subsequent mask
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    # combine padded & subsequent mask
    tgt_mask = tgt_mask & tgt_nopeak_mask
    return tgt_mask

# finds maxlen between inputs and labels
def get_maxlen(inputs, labels):
    return max(len(max(inputs, key=len)), len(max(labels, key=len)))

# loads a dataset from huggingface hub
def load(*args, **kwargs):
    return load_dataset(*args, **kwargs)

class Dataset(IterableDataset):

    def __init__(self, inputs, labels):
        if len(inputs) != len(labels):
            raise ValueError(f"the size of inputs ({len(inputs)}) \
must match the size of labels ({len(labels)})")
        if type(inputs[0]) != type(labels[0]):
            raise ValueError(f"the data type of inputs ({type(inputs[0])}) \
must match the data type of labels ({type(labels[0])})")
        self.inputs = inputs
        self.labels = labels
        self.size = len(inputs)
        self.dtype = type(inputs[0])

    # returns data as list
    def list(self):
        if isinstance(self.inputs, torch.Tensor) or isinstance(self.inputs, np.ndarray):
            return self.inputs.tolist(), self.labels.tolist()
        return self.inputs, self.labels

    # returns data as tensors
    def tensors(self):
        if self.dtype == str or self.dtype == np.str_:
            raise TypeError(f"torch does not support type {self.dtype}")
        if isinstance(self.inputs, np.ndarray):
            return torch.from_numpy(self.inputs), torch.from_numpy(self.labels)
        return torch.tensor(self.inputs), torch.tensor(self.labels)

    # returns data as numpy array
    def numpy(self):
        if isinstance(self.inputs, torch.Tensor):
            return self.inputs.numpy(), self.labels.numpy()
        return np.array(self.inputs), np.array(self.labels)

    # returns the entire dataset as a corpus
    def corpus(self):
        # combine correct type object
        if isinstance(self.inputs, np.ndarray):
            return np.append(self.inputs, self.labels)
        elif isinstance(self.inputs, torch.Tensor):
            return torch.cat((self.inputs, self.labels), dim=0)
        return self.inputs + self.labels

    def sample(self):
        index = np.random.randint(self.size)
        return self[index]

    # returns a sampled batch of specified batch size
    def batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError(f"cannot sample batch larger than size ({self.size})")
        # get random slices
        indices = np.random.choice(len(self), (batch_size, ), replace=False)
        inputs, labels = self.numpy()
        input_samples, label_samples = inputs[indices], labels[indices]
        
        # return correct list object
        if isinstance(self.inputs, torch.Tensor):
            return torch.from_numpy(input_samples), torch.from_numpy(label_samples)
        elif isinstance(self.inputs, list):
            return input_samples.tolist(), label_samples.tolist()
        return input_samples, label_samples

    # returns a dataframe of data
    def dataframe(self, headers=None):
        if headers is None:
            headers = ("inputs", "labels")
        d = {f"{headers[0]}": self.inputs, f"{headers[1]}": self.labels}
        return pd.DataFrame(d)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __iter__(self):
        return iter([(input, label) for input, label in zip(self.inputs, self.labels)])

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.dataframe())

    # gives back a tokenized dataset
    def tokenized(self, tokenizer, model=True):
        inputs, labels = self.list()
        input_tokens, label_tokens = tokenizer.encode(inputs, model=model), \
            tokenizer.encode(labels, model=model)
        return Dataset(input_tokens, label_tokens)

    def dataloader(self, batch_size=32, shuffle=False, drop_last=True, **dataloader_kwargs):
        # create tensors
        inputs, labels = self.tensors()
        tensorset = TensorDataset(inputs, labels)
        # create dataloader with specified args
        dataloader = DataLoader(tensorset, batch_size=batch_size, shuffle=shuffle, 
                                drop_last=drop_last, **dataloader_kwargs)
        return dataloader

if __name__ == "__main__":
    pass

    