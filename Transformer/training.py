import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks

class Scheduler:

    def __init__(self, optimizer, dm, warmup):
        self.optimizer = optimizer
        self.dm = dm
        self.warmup = warmup
        self.steps = 1

    def step(self):
        # calculate the learning rate
        steps = self.steps
        groups = self.optimizer.param_groups
        lr = np.power(self.dm, -0.5) * \
            min(np.power(steps, -0.5), steps * np.power(self.warmup, -1.5))
        print(lr)
        # apply the learning rate to optimizer
        for group in groups:
            group["lr"] = lr
        self.steps += 1 # update num of steps

def train(model, optimizer, scheduler, dataloader, epochs=5, verbose=True, device=None):

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
        scheduler.step() # set current lr w/ scheduler

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

            if verbose:
                if (i + 1) % int(m * verbose) == 0 and (i + 1) != m:
                    # diplay info every 25% of an epoch
                    print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")
        net_loss += accum_loss / m
        # display info after end of epoch
        if verbose:
            print(f"Epoch {epoch + 1} Complete | Epoch Average Loss: {accum_loss / m:.4f}")
    net_loss /= epochs
    # display info after end of training
    if verbose:
        print(f"Training Complete | Overall Average Loss: {net_loss:.4f}")
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
    predictions = tokenizer.decode(predictions, special_tokens=False)
    outputs = []
    # combine seq & predictions
    for seq, pred in zip(sequences, predictions):
        outputs.append(f"{seq} -> {pred}")
    return outputs

def prompt(model, tokenizer, start, end, maxlen=None, device=None):

    # default
    if maxlen is None:
        maxlen = 25

    # inference
    model.eval()
    tokenizer.inference()
    softmax = nn.Softmax(dim=-1)

    # get input and tokenize
    sequence = input("Enter in the sequence of text:\n\n").strip()
    ids = tokenizer.encode(sequence, model=True)

    # create src & tgt tensor
    src = torch.tensor(ids).unsqueeze(0).long().to(device)
    tgt = torch.tensor([start]).unsqueeze(0).to(device)

    # predict start from src until maxlen or end token hit
    while tgt.size(1) <= maxlen:  

        # get model output 
        out = model(src, tgt, src_mask=None, tgt_mask=None)
        # get probability distribution
        prob = softmax(out)

        # get token with highest probability
        pred = torch.argmax(prob, dim=-1)[:, -1]
        pred = pred.contiguous().view(-1, 1)

        # combine prediction
        tgt = torch.cat((tgt, pred), dim=-1)

        # predicted end
        if pred.item() == end:
            break

    # maxlen exceeded
    return f"{sequence} -> {tokenizer.decode(tgt.tolist(), special_tokens=False)[0]}"

if __name__ == "__main__":
    pass
    
