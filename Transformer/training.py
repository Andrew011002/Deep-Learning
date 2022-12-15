import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks


class Checkpoint:

    def __init__(self, model, optimizer, scheduler, checkpoch, path="", overwrite=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoch = checkpoch
        self.path = path
        self.overwrite = overwrite
        self.epoch = 0

    def check(self):
        # save current model state
        if (self.epoch + 1) % self.checkpoch == 0:
            self.save()
        self.epoch += 1 # update steps taken

    def save(self):
        # save same path
        if self.overwrite:
            path = f"{self.path}checkpoint.pt"
        # save diff path
        else:
            path = f"{self.path}checkpoint-{self.epoch + 1}.pt"

        # save params of
        torch.save({
            "model_params": self.model.state_dict(),
            "optimizer_params": self.optimizer.state_dict(),
            "scheduler_params": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "checkpoch":self.checkpoch
        }, path)
        print(f"Model, optimizer, scheduler, and checkpoint saved")
            
    # loads model, optimizer, and scheduler from save
    def load(self, model, optimizer, scheduler, path, device=None):
        checkpoint = torch.load(path, map_location=device) # load check point
        # apply to all modules
        model = model.load_state_dict(checkpoint["model_params"])
        optimizer = optimizer.load_state_dict(checkpoint["optimizer_params"])
        scheduler = scheduler.load_state_dict(checkpoint["scheduler_params"])
        epoch = checkpoint["epoch"]
        checkpoch = checkpoint["checkpoch"]
        # restore checkpoint
        checkpoint = Checkpoint(model, optimizer, scheduler, checkpoch, self.path, self.overwrite)
        checkpoint.epoch = epoch
        print(f"Model, optimizer, scheduler, and epoch loaded")
        return checkpoint

def train(model, optimizer, dataloader, epochs=5, warmups=100, scheduler=None, 
            checkpoint=None, verbose=True, device=None):

    if verbose:
        print("Training Started")
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    model.train()
    m = len(dataloader)
    net_loss = 0

    for epoch in range(epochs):
        
        # reset accumulative loss and display current epoch
        if verbose:
            print(f"Epoch {epoch + 1} Started")
        accum_loss = 0

        for i, data in enumerate(dataloader):
            # get source and targets
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape src: (batch_size, srclen) tgt & out: (batch_size, outlen)
            src, tgt, out = src.long(), tgt.long(), out.long()
            # generate the mask
            src_mask, tgt_mask = generate_masks(src, tgt, model.pad_id)
            
            # move to device
            src, tgt, out = src.to(device), tgt.to(device), out.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # zero the gradient
            optimizer.zero_grad()
            # get prediction and reshape outputs
            pred = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # shape: (batch_size, seq_len, vocab_size)
            pred, out = pred.view(-1, pred.size(-1)), out.contiguous().view(-1) # shape pred: (batch_size * seq_len, vocab_size) out: (batch_size * seq_len)
            # calculate loss and backpropagate
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # tally loss over time
            accum_loss += loss.item()

            # check on checkpoint
            if checkpoint:
                checkpoint.check()

            # diplay info every fraction of an epoch
            if verbose:
                if (i + 1) % int(m * verbose) == 0 and (i + 1) != m:
                    print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")

        net_loss += accum_loss / m
        # apply scheduler after warmups
        if epoch + 1 > warmups and scheduler:
            scheduler.step(accum_loss / m) 
        # display info after end of epoch
        if verbose:
            print(f"Epoch {epoch + 1} Complete | Epoch Loss: {accum_loss / m:.4f}")

    net_loss /= epochs # avg accum loss over epochs
    # display info after end of training
    if verbose:
        print(f"Training Complete | Training Loss: {net_loss:.4f}")
    return net_loss

def predict(sequences, model, tokenizer, start, end, maxlen, device=None):
    # sequence inshape: (batch_size, src_len,)

    # inference
    tokenizer.inference()
    token_ids = tokenizer.encode(sequences, model=True)
    softmax = nn.Softmax(dim=-1)
    model.eval()

    # get prediction for each sequence
    predictions = []
    for ids in token_ids:
        # create src tensor
        ids = np.array(ids, dtype=int)
        src = torch.from_numpy(ids).unsqueeze(0).long() # (unknown, src_len)
        src_mask = (src != model.pad_id).unsqueeze(-2)

        # create tgt tensor
        tgt = torch.tensor([start]).unsqueeze(0).long() # generate start
        
        # move tensors to device
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt = tgt.to(device)

        # predict one token at a time
        while tgt.size(1) < maxlen:
            # get model output
            out = model(src, tgt, src_mask=src_mask)
            # get probability distribution
            prob = softmax(out)

            # get last token(s) of highest probability
            pred = torch.argmax(prob, dim=-1)[:, -1] # shape: (batch_size, 1)
            pred = pred.contiguous().view(-1, 1)
            # add token to current tgt (batch_size, output_size + 1)
            tgt = torch.cat((tgt, pred), dim=-1)

            # done prediction
            if pred.item() == end:
                break
        # store tokens of predictions
        predictions.append(tgt.squeeze().tolist())

    # create continuations
    predictions = tokenizer.decode(predictions)
    outputs = []
    # combine seq & predictions
    for seq, pred in zip(sequences, predictions):
        outputs.append(f"{seq} -> {pred}")
    return outputs

def prompt(model, tokenizer, start, end, device=None):
    # default

    # inference
    model.eval()
    tokenizer.inference()
    softmax = nn.Softmax(dim=-1)

    # get input and tokenize
    sequence = input("Enter in the sequence of text:\n\n").strip()
    ids = tokenizer.encode(sequence, model=True)
    maxlen = len(ids[0])

    # create src & tgt tensor
    src = torch.tensor(ids).unsqueeze(0).long().to(device)
    tgt = torch.tensor([start]).unsqueeze(0).to(device)

    # predict start from src until maxlen or end token hit
    while tgt.size(1) <= maxlen:  

        # get model output 
        out = model(src, tgt, src_mask=None, tgt_mask=None)
        # get probability distribution
        prob = softmax(out)
        print(prob)

        # get token with highest probability
        pred = torch.argmax(prob, dim=-1)[:, -1]
        pred = pred.contiguous().view(-1, 1)

        # combine prediction
        tgt = torch.cat((tgt, pred), dim=-1)

        # predicted end
        if pred.item() == end:
            break

    # maxlen exceeded
    return f"{sequence} -> {tokenizer.decode(tgt.tolist())[0]}"

if __name__ == "__main__":
    import torchvision
    model = torchvision.models.vgg16()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    epochs = 100
    checkpoint = Checkpoint(model, optimizer, scheduler, 25)
    for epoch in range(epochs):
        checkpoint.check()
    checkpoint = Checkpoint().load(model, optimizer, scheduler, path="checkpoint-25.pt")

    
