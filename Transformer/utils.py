import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

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

# splits sequence pairs up to a desired size from a dataset
def get_split(datadict, input_key, label_key, size=100000):
    inputs, labels = [], []
    count = 0
    # add sequence pairs while the desired size isn't reached
    for pair in datadict["translation"]:
        if pair[input_key] != pair[label_key]:
            inputs.append(pair[input_key])
            labels.append(pair[label_key])
            count += 1
        # size reached
        if count == size:
            break
    return inputs, labels

class Dataset(IterableDataset):

    def __init__(self, inputs, labels):
        if len(inputs) != len(labels):
            raise ValueError(f"The size of inputs ({len(inputs)}) \
must match the size of labels ({len(labels)})")
        if type(inputs[0]) != type(labels[0]):
            raise ValueError(f"The data type of inputs ({type(inputs[0])}) \
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
            raise TypeError(f"PyTorch does not support type {self.dtype}")
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
        # combine inputs and labels
        inputs, labels = self.list()
        return inputs + labels

    # returns a sampled batch of specified batch size
    def sample(self, n=1):
        if n > self.size:
            raise ValueError(f"Cannot sample batch larger than size ({self.size})")
        # get random slices
        indices = np.random.choice(len(self), (n, ), replace=False).astype(int)
        inputs, labels = self.inputs, self.labels
        # return as pairs
        return [(inputs[i], labels[i]) for i in indices]

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

    # finds the average length of tokenized sequences
    def avglen(self, tokenizer, factor=1):
        inputs, labels = self.list()
        # give back higher average of average lengths of sequences
        m = sum(len(input) for input in tokenizer(inputs, model=True)) / self.size
        n = sum(len(input) for input in tokenizer(labels, model=True)) / self.size
        # apply factor to incorporate outlier sequences
        return int(np.rint(max(m, n)) * factor)
    
    # gives back maxlen between tokenized sequences
    def maxlen(self, tokenizer, factor=1):
        inputs, labels = self.list()
        # find max longest sequence in inputs & labels
        max_inputs = len(max(tokenizer(inputs), key=len))
        max_labels = len(max(tokenizer(labels), key=len))
        # give back greatest of the two
        return int(np.rint(max(max_inputs, max_labels) * factor))

    def dataloader(self, batch_size=32, shuffle=False, drop_last=True, **dataloader_kwargs):
        # create tensors
        inputs, labels = self.tensors()
        tensorset = TensorDataset(inputs, labels)
        # create dataloader with specified args
        dataloader = DataLoader(tensorset, batch_size=batch_size, shuffle=shuffle, 
                                drop_last=drop_last, **dataloader_kwargs)
        return dataloader

class Checkpoint:

    def __init__(self, model, optimizer, scheduler=None, evaluator=None, clock=None, epochs=5, path=None, overwrite=False, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.clock = clock
        self.epochs = epochs
        self.path = path
        self.overwrite = overwrite
        self.epoch = 0
        self.loss = None
        self.bleu = None
        self.verbose = verbose

    def check(self, loss):
        # save current model state
        self.loss = loss
        self.epoch += 1 # update steps taken
        if self.epoch % self.epochs == 0:
            self.save()
            return True
        return False

    def save(self):
        # give name
        if self.path is None:
            self.path = "checkpoint"
        # create path if not existent 
        create_path(self.path)

        # save same path
        if self.overwrite:
            path = f"{self.path}.pt"
        # save diff path
        else:
            path = f"{self.path}-{self.epoch}.pt"

        # save params of
        torch.save({
            # required
            "model_params": self.model.state_dict(),
            "optimizer_params": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "epochs": self.epochs,
            "loss": self.loss,
            "path": self.path,
            "overwrite": self.overwrite,
            # optional
            "scheduler_params": self.scheduler.state_dict() if self.scheduler \
                else None,
            "evaluator": self.evaluator if self.evaluator \
                else None,
            "bleu": self.evaluator.bleu if self.evaluator \
                else None,
            "duration": self.clock.duration if self.clock \
                else None
        }, path)
        # display info
        if self.verbose:
            print(f"Checkpoint saved")
            
    def load_checkpoint(self, path=None, device=None): 
        if path is None:
            path = "checkpoint"
        # load checkpoint
        path = f"{path}.pt"
        checkpoint = torch.load(path, map_location=device) 
        # overwrite current state of checkpoint (required)
        self.model.load_state_dict(checkpoint["model_params"])
        self.optimizer.load_state_dict(checkpoint["optimizer_params"])
        self.epoch = checkpoint["epoch"]
        self.epochs = checkpoint["epochs"]
        self.loss = checkpoint["loss"]
        self.path = checkpoint["path"]
        self.overwrite = checkpoint["overwrite"]
        # overwrite current state of checkpoint (optional)
        if checkpoint["scheduler_params"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_params"])
        if checkpoint["evaluator"] and self.evaluator:
            self.evaluator = checkpoint["evaluator"]
        if checkpoint["bleu"] and self.bleu:
            self.bleu = checkpoint["bleu"]
        if checkpoint["duration"] and self.clock:
            self.clock = Clock(checkpoint["duration"])
    
        # display info
        if self.verbose:
            print(f"Checkpoint loaded")
    
    def state_dict(self):
        return {"model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "evaluator": self.evaluator,
                "epoch": self.epoch,
                "loss": self.loss,
                "bleu": self.bleu,
                "clock": self.clock
                }

class Clock:

    def __init__(self, duration=0) -> None:
        self.duration = duration
        self.zero = None
        self.current = None

    def start(self):
        self.zero = time.time()
        self.current = self.zero

    def tick(self):
        now = time.time()
        elapsed = now - self.zero
        self.duration += elapsed

    def clock(self, start, end):
        elapsed = end - start
        self.current = end
        return self.to_hour_min_sec(elapsed)
    
    def epoch(self):
        now = time.time()
        h, m, s = self.clock(self.current, now)
        return self.asstr(h, m, s)

    def elapsed(self):
        self.tick()
        elapsed = self.duration
        h, m, s = self.to_hour_min_sec(elapsed)
        return self.asstr(h, m, s)
        
    def reset(self):
        self.__init__()

    def to_hour_min_sec(self, elapsed):
        hours, rem = elapsed // 3600, elapsed % 3600
        minutes, seconds = rem // 60, rem % 60
        return hours, minutes, seconds
    
    def asstr(self, hours, minutes, seconds):
        hstr = f"0{hours:.0f}" if hours < 10 else f"{hours:.0f}"
        mstr = f"0{minutes:.0f}" if minutes < 10 else f"{minutes:.0f}"
        sstr = f"0{seconds:.0f}" if seconds < 10 else f"{seconds:.0f}"
        return f"{hstr}:{mstr}:{sstr}"

# saves the model to a path
def save_model(model, path=None):
    # default
    path = "model" if path is None else path

    # create path if non-existant
    create_path(path)
    
    # save model to the path
    torch.save(model.state_dict(), f"{path}.pth")
    print(f"Model params saved")

# loads a model params from a path
def load_model(model, path=None, device=None):
    # default
    path = "model" if path is None else path

    # load parameters into model
    params = torch.load(f"{path}.pth", map_location=device)
    if torch.cuda.is_available():
        model.load_state_dict(params)
    print(f"Model params loaded")
    return model

# creates a path if it doesn't exist
def create_path(path):
    path = path.split("/")
    path = "/".join(path[:-1]) + "/"
    if path and not os.path.exists(path):
        os.makedirs(path)
    
if __name__ == "__main__":
    pass

    
    
    