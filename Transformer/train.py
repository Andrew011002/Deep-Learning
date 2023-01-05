import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks

def train(dataloader, model, optimizer, scheduler=None, evaluator=None, 
    checkpoint=None, clock=None, epochs=1000, warmups=100, verbose=True, device=None):
    # setup
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    m = len(dataloader)
    net_loss = curr_bleu = bleu = 0
    saved = done = False
    model.train()
    if clock:
        clock.reset()
        clock.start()
    if verbose:
        print("Training Started")

    # train over epochs
    for epoch in range(epochs):

        accum_loss = 0 # reset accumulative loss

        for i, data in enumerate(dataloader):
            # get src & tgt
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape: src - (batch_size, src_len) tgt & out - (batch_size, tgt_len)
            src, tgt, out = src.long(), tgt.long(), out.long()
            # generate the masks
            src_mask, tgt_mask = generate_masks(src, tgt, model.pad_id)
            # move to device 
            src, tgt, out = src.to(device), tgt.to(device), out.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # zero the gradient
            optimizer.zero_grad()
            # get pred & reshape outputs
            pred = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # shape: pred - (batch_size, seq_len, vocab_size)
            pred, out = pred.view(-1, pred.size(-1)), out.contiguous().view(-1) # shape: pred - (batch_size * seq_len, vocab_size) out - (batch_size * seq_len)
            # calc loss & backprop
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # accumulate loss over time
            accum_loss += loss.item()

        # get losses
        epoch_loss = accum_loss / m
        net_loss += epoch_loss
        # apply scheduler after warmups (if applicable)
        warmup = epoch + 1 < warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(net_loss)
        # evaluate model (if applicable)
        if evaluator:
            curr_bleu = evaluator.evaluate(model)
            done = evaluator.done()
            model.train() # reset back (model.eval() called)
            bleu = evaluator.bleu
        if verbose:
            train_printer(epoch_loss, epoch + 1, clock, curr_bleu, warmup, saved)
        # model meets bleu score (complete training)
        if done:
            break

    # calc avg train loss
    train_loss = net_loss / epochs 
    if verbose:
        train_printer(train_loss, None, clock, bleu, None, saved)
    return train_loss

def retrain(checkpoint, epochs=1000, warmups=100, verbose=True, device=None):
    # grab info from checkpoint
    info = checkpoint.state_dict()
    dataloader = info["dataloader"]
    model = info["model"]
    optimizer = info["optimizer"]
    scheduler = info["scheduler"]
    evaluator = info["evaluator"]
    clock = info["clock"]
    epoch_start = info["epoch"]
    net_loss = info["loss"]
    bleu = info["bleu"]

     # setup
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    m = len(dataloader)
    curr_bleu = 0
    saved = done = False
    model.train()
    if clock:
        clock.start()
    if verbose:
        print("Training Resumed")

    # train over epochs
    for epoch in range(epoch_start, epochs):

        accum_loss = 0 # reset accumulative loss

        for i, data in enumerate(dataloader):
            # get src & tgt
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape: src - (batch_size, src_len) tgt & out - (batch_size, tgt_len)
            src, tgt, tgt = src.long(), tgt.long(), out.long()
            # generate the mask
            src_mask, tgt_mask = generate_masks(src, tgt, model.pad_id)
            # move to device
            src, tgt, out = src.to(device), tgt.to(device), out.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # zero the gradient
            optimizer.zero_grad()
            # get pred & reshape outputs
            pred = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # shape: pred - (batch_size, seq_len, vocab_size)
            pred, out = pred.view(-1, pred.size(-1)), out.contiguous().view(-1) # shape: pred - (batch_size * seq_len, vocab_size) out - (batch_size * seq_len)
            # calc loss & backprop
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # accumulate loss over time
            accum_loss += loss.item()

        # get losses
        epoch_loss = accum_loss / m
        net_loss += epoch_loss
        # apply scheduler after warmups (if applicable)
        warmup = epoch + 1 < warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(net_loss)
        # evaluate model (if applicable)
        if evaluator:
            curr_bleu = evaluator.evaluate(model)
            done = evaluator.done()
            model.train() # reset back (model.eval() called)
            bleu = evaluator.bleu

        # display info after end of epoch
        if verbose:
            train_printer(epoch_loss, epoch + 1, clock, curr_bleu, warmup, saved)
        # model meets bleu score (complete training)
        if done:
            break
    
    # calc avg train loss
    train_loss = net_loss / epochs 
    if verbose:
        train_printer(train_loss, None, clock, bleu, None, saved)
    return train_loss

def train_printer(loss, epoch=None, clock=None, bleu=None, warmup=None, saved=None):
    # basic info
    div = f"{'-' * 79}"
    info = f"Epoch {epoch} Complete | " if epoch else "Training Complete | "
    # time info
    if clock:
        info += f"Epoch Duration: {clock.epoch()} | " if epoch else ""
        info += f"Elapsed Training Time: {clock.elapsed()} |"
    info += "\n"
    # metrics
    info += f"Metrics | Epoch Loss: {loss:.4f} | " if epoch else f"Metrics | Training Loss: {loss:.4f} | "
    if bleu is not None:
        info += f"BLEU Score: {bleu:.1f} | " if epoch else f"Best BLEU: {bleu:.1f} | "
    info += "\n"
    info += "Other Info | "
    # other info
    if warmup is not None or saved is not None:
        # warmup step
        info += f"Scheduler Warmup Step: {warmup} | " if warmup is not None else ""
        # checkpoint
        info += f"Checkpoint Saved: {saved} |" if saved is not None else ""
    # no other info
    else:
        info += "NA |"
    print(div)
    print(info)



if __name__ == "__main__":
    pass

    

    
