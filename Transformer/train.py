import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks, write

def train(dataloader, model, optimizer, scheduler=None, evaluator=None, checkpoint=None, clock=None, 
        epochs=1000, warmups=100, verbose=True, log=None, device=None):
    # setup
    m = len(dataloader)
    done = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    losses, bleus = [], []
    saved = bleu = None
    if clock:
        clock.reset()
        clock.start()
    if verbose:
        output = f"{'-' * 79}\nTraining Started"
        print(output)
        if log is not None:
            write(output, log, overwrite=True)

    # train over epochs
    for epoch in range(epochs):
        model.train() # set to train (set to eval for evaluator)
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

        # get losses & keep track
        epoch_loss = accum_loss / m
        losses.append(epoch_loss)
        # apply scheduler after warmups (if applicable)
        warmup = epoch + 1 <= warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        if evaluator:
            bleu = evaluator.evaluate()
            bleus.append(bleu)
            done = evaluator.done()
        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(losses, bleus)
        # evaluate model (if applicable)
        if verbose:
            output = train_printer(epoch_loss, epoch + 1, clock, bleu, warmup, saved)
            # write to log file (if applicable)
            if log:
                write(output, log, overwrite=False)
        # model meets bleu score (complete training)
        if done:
            break

    # calc avg train loss & best bleu (if applicable)
    net_loss = np.mean(losses).item()
    best_bleu = max(bleus) if bleus else None
    if verbose:
        output = train_printer(net_loss, None, clock, best_bleu, None, saved)
        # write to log file (if applicable)
        if log:
            write(output, log, overwrite=False)
    return losses, bleus

def retrain(checkpoint, epochs=1000, warmups=100, verbose=True, log=None, device=None):
    # grab info from checkpoint
    info = checkpoint.state_dict()
    dataloader = info["dataloader"]
    model = info["model"]
    optimizer = info["optimizer"]
    scheduler = info["scheduler"]
    evaluator = info["evaluator"]
    clock = info["clock"]
    epoch_start = info["epoch"]
    losses = info["losses"]
    bleus = info["bleus"]

     # setup
    m = len(dataloader)
    done = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    saved = bleu = None
    if clock:
        clock.start()
    if verbose:
        output = f"{'-' * 79}\nTraining Resumed"
        print(output)
        if log is not None:
            write(output, log, overwrite=False)

    # train over epochs
    for epoch in range(epoch_start, epochs):
        model.train() # set to train (set to eval for evaluator)
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

        # get losses & keep track
        epoch_loss = accum_loss / m
        losses.append(epoch_loss)
        # apply scheduler after warmups (if applicable)
        warmup = epoch + 1 <= warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        # evaluate model (if applicable)
        if evaluator:
            bleu = evaluator.evaluate()
            bleus.append(bleu)
            done = evaluator.done()
        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(losses, bleus)
        # display info after end of epoch
        if verbose:
            output = train_printer(epoch_loss, epoch + 1, clock, bleu, warmup, saved)
            # write to log file (if applicable)
            if log:
                write(output, log, overwrite=False)
        # model meets bleu score (complete training)
        if done:
            break
    
    # calc avg train loss & best bleu (if applicable)
    net_loss = np.mean(losses).item()
    best_bleu = max(bleus) if bleus else None
    if verbose:
        output = train_printer(net_loss, None, clock, best_bleu, None, saved)
        # write to log file (if applicable)
        if log:
            write(output, log, overwrite=False)
    return losses, bleus

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
    
    # generate final output
    output = f"{div}\n{info}"
    print(output)
    return output
    
    

if __name__ == "__main__":
    pass

    

    
