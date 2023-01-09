import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

def parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6

def model_size(model):
    params = model.parameters()
    buffers = model.buffers()
    param_size_mb = sum(param.nelement() * param.element_size() for param in params)
    buffer_size_mb = sum(buffer.nelement() * buffer.element_size() for buffer in buffers)
    return (param_size_mb + buffer_size_mb) / np.power(1024, 2)

def generate_masks(src, tgt, pad_id):
    # inshape: src - (batch_size, src_len) tgt - (batch_size, tgt_len)

    # create padded mask for src (True = no pad False = pad) shape: src_mask - (batch_size, 1, src_len) 
    src_mask = (src != pad_id).unsqueeze(-2)

    # generate padded nopeak mask for tgt
    tgt_mask = generate_nopeak_pad_mask(tgt, pad_id)
    return src_mask, tgt_mask

def generate_nopeak_pad_mask(tgt, pad_id):
    # inshape: tgt - (batch_size, tgt_len)

    # create padded mask for src (True = no pad False = pad) shape: tgt_mask - (batch_size, 1, tgt_len)
    tgt_mask = (tgt != pad_id).unsqueeze(-2)
    # create subsequent mask shape: tgt_nopeak_mask - (1, tgt_len, tgt_len)
    tgt_len = tgt.size(1)
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    # combine padded & subsequent mask shape: tgt_mask - (batch_size, tgt_len, tgt_len)
    tgt_mask = tgt_mask & tgt_nopeak_mask
    return tgt_mask

def get_split(datadict, input_key, label_key, size=100000):
    # assort through all pairs until size size is met or end
    inputs, labels = set(), set()
    count = 0

    for pair in datadict["translation"]:
        input, label = pair[input_key], pair[label_key]
        # add non-duplicates & unique pairs
        if input != label and input not in inputs and label not in labels:
            inputs.add(pair[input_key])
            labels.add(pair[label_key])
            count += 1
        # size reached
        if count == size:
            break
    return list(inputs), list(labels)

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

    def list(self):
        if isinstance(self.inputs, torch.Tensor) or isinstance(self.inputs, np.ndarray):
            return self.inputs.tolist(), self.labels.tolist()
        return self.inputs, self.labels

    def tensors(self):
        if self.dtype == str or self.dtype == np.str_:
            raise TypeError(f"PyTorch does not support type {self.dtype}")
        if isinstance(self.inputs, np.ndarray):
            return torch.from_numpy(self.inputs), torch.from_numpy(self.labels)
        return torch.tensor(self.inputs), torch.tensor(self.labels)

    def numpy(self):
        if isinstance(self.inputs, torch.Tensor):
            return self.inputs.numpy(), self.labels.numpy()
        return np.array(self.inputs), np.array(self.labels)

    def corpus(self, data=None):
        if data != "inputs" and data != "labels" and data is not None:
            raise ValueError(f"Unknown argument for data: {data}\
Valid arguments: (inputs, labels, both)")

        # return the proper corpus
        data = "both" if data is None else data
        inputs, labels = self.list()
        if data == "inputs":
            return inputs
        if data == "labels":
            return labels
        if data == "both":
            return inputs + labels

    def sample(self, n=1):
        if n > self.size:
            raise ValueError(f"Cannot sample batch larger than size ({self.size})")

        # get random slices
        indices = np.random.choice(len(self), (n, ), replace=False).astype(int)
        inputs, labels = self.inputs, self.labels
        # return as pairs
        return [(inputs[i], labels[i]) for i in indices]

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

    def tokenized(self, tokenizer, model=True):
        # encode inputs & labels
        inputs, labels = self.list()
        input_tokens, label_tokens = tokenizer.encode(inputs, model=model, module="encoder"), \
            tokenizer.encode(labels, model=model, module="decoder")
        # return as Dataset
        return Dataset(input_tokens, label_tokens)

    def avglen(self, tokenizer, factor=1):
        # find max between avg input length & avg label length
        inputs, labels = self.list()
        m = sum(len(input) for input in \
            tokenizer.encode(inputs, model=True, module="encoder")) / self.size
        n = sum(len(label) for label in \
            tokenizer.encode(labels, model=True, module="decoder")) / self.size
        # apply factor to incorporate outlier sequences
        return int(np.rint(max(m, n)) * factor)
    
    def maxlen(self, tokenizer, factor=1):
        inputs, labels = self.list()
        # find max longest sequence in inputs & labels
        max_inputs = len(max(tokenizer.encode(inputs, module="encoder"), key=len))
        max_labels = len(max(tokenizer.encode(labels, module="decoder"), key=len))
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

    def __init__(self, dataloader, model, optimizer, scheduler=None, evaluator=None, 
                    clock=None, epochs=5, path=None, overwrite=False):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.clock = clock
        self.epochs = epochs
        self.path = path
        self.overwrite = overwrite
        self.epoch = 0
        self.losses = None
        self.bleus = None
        

    def check(self, losses, bleus):
        # save current model state
        self.losses = losses
        self.bleus = bleus
        self.epoch += 1 # update steps taken
        if self.epoch % self.epochs == 0:
            self.save()
            return True
        return False

    def save(self):
        # default
        if self.path is None:
            self.path = "checkpoint"
        # create path (if non-existent)
        create_path(self.path)
        # save same path
        if self.overwrite:
            path = f"{self.path}.pt"
        # save diff path
        else:
            path = f"{self.path}-{self.epoch}.pt"

        torch.save({
            # required
            "dataloader": self.dataloader,
            "model_params": self.model.state_dict(),
            "optimizer_params": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "epochs": self.epochs,
            "losses": self.losses,
            "path": self.path,
            "overwrite": self.overwrite,
            # optional
            "scheduler_params": self.scheduler.state_dict() if self.scheduler \
                else None,
            "evaluator": self.evaluator if self.evaluator \
                else None,
            "bleus": self.bleus if self.bleus \
                else None,
            "duration": self.clock.duration if self.clock \
                else None
        }, path)
            
    def load_checkpoint(self, path=None, verbose=True, device=None): 
        # default
        if path is None:
            path = "checkpoint"
        # load checkpoint
        path = f"{path}.pt"
        checkpoint = torch.load(path, map_location=device) 
        # overwrite current state of checkpoint (required)
        self.dataloader = checkpoint["dataloader"]
        self.model.load_state_dict(checkpoint["model_params"])
        self.optimizer.load_state_dict(checkpoint["optimizer_params"])
        self.epoch = checkpoint["epoch"]
        self.epochs = checkpoint["epochs"]
        self.losses = checkpoint["losses"]
        self.path = checkpoint["path"]
        self.overwrite = checkpoint["overwrite"]
        # overwrite current state of checkpoint (optional)
        if checkpoint["scheduler_params"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_params"])
        if checkpoint["evaluator"]:
            self.evaluator = checkpoint["evaluator"]
        if checkpoint["bleus"]:
            self.bleus = checkpoint["bleus"]
        if checkpoint["duration"] is not None:
            self.clock = Clock(checkpoint["duration"])
        if verbose:
            print("Checkpoint Loaded")
    
    def state_dict(self):
        return {"dataloader": self.dataloader,
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "evaluator": self.evaluator,
                "epoch": self.epoch,
                "losses": self.losses,
                "bleus": self.bleus,
                "clock": self.clock
                }

class Clock:

    def __init__(self, duration=0) -> None:
        self.duration = duration
        self.current = None

    def start(self):
        self.current = time.time()

    def clock(self, start, end):
        elapsed = end - start
        self.current = end
        self.duration += elapsed
        return self.to_hour_min_sec(elapsed)
    
    def epoch(self):
        now = time.time()
        h, m, s = self.clock(self.current, now)
        return self.asstr(h, m, s)

    def elapsed(self):
        elapsed = self.duration
        h, m, s = self.to_hour_min_sec(elapsed)
        return self.asstr(h, m, s)
        
    def reset(self):
        self.__init__(duration=0)

    def to_hour_min_sec(self, elapsed):
        hours, rem = elapsed // 3600, elapsed % 3600
        minutes, seconds = rem // 60, rem % 60
        return hours, minutes, seconds
    
    def asstr(self, hours, minutes, seconds):
        hstr = f"0{hours:.0f}" if np.rint(hours) < 10 else f"{hours:.0f}"
        mstr = f"0{minutes:.0f}" if np.rint(minutes) < 10 else f"{minutes:.0f}"
        sstr = f"0{seconds:.0f}" if np.rint(seconds) < 10 else f"{seconds:.0f}"
        return f"{hstr}:{mstr}:{sstr}"

def save_model(model, path=None):
    # default
    path = "model" if path is None else path
    # create path (if non-existent)
    create_path(path)
    # save model to the path
    torch.save(model.state_dict(), f"{path}.pth")
    print(f"Model params saved")

def load_model(model, path=None, device=None):
    # default
    path = "model" if path is None else path
    # load parameters into model
    params = torch.load(f"{path}.pth", map_location=device)
    if torch.cuda.is_available():
        model.load_state_dict(params)
    print(f"Model params loaded")
    return model

def create_path(path):
    path = path.split("/")
    path = "/".join(path[:-1]) + "/"
    if path and not os.path.exists(path):
        os.makedirs(path)

def write(contents, path, overwrite=False):
    create_path(path)
    arg = "w" if overwrite else "a"
    file = open(path, arg)
    file.write(contents)
    file.close()

    
if __name__ == "__main__":
    pass

    
    
    