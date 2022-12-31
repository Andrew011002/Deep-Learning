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
            # get src and tgt
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape: src (batch_size, src_len) tgt & out (batch_size, out_len)
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
            # calculate loss and back-propagate
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # tally loss over time
            accum_loss += loss.item()

        # get losses
        epoch_loss = accum_loss / m
        net_loss += epoch_loss
        # apply scheduler after warmups
        warmup = epoch + 1 < warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        # check on checkpoint
        if checkpoint:
            saved = checkpoint.check(epoch_loss)
        # evaluate model
        if evaluator:
            curr_bleu = evaluator.evaluate(model)
            done = evaluator.done()
            model.train() # reset back
            bleu = evaluator.bleu

        # display info after end of epoch
        if verbose:
            train_printer(epoch_loss, epoch + 1, clock, curr_bleu, warmup, saved)
        # model meets bleu score
        if done:
            break

    net_loss /= epochs # avg accum loss over epochs
    # display info after end of training
    if verbose:
        train_printer(net_loss, None, clock, bleu, None, saved)
    return net_loss

def retrain(dataloader, checkpoint, epochs=1000, warmups=100, verbose=True, device=None):
    
    # resume from checkpoint
    info = checkpoint.state_dict()
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
            # get src and tgt
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape: src (batch_size, src_len) tgt & out (batch_size, out_len)
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
            # calculate loss and back-propagate
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # tally loss over time
            accum_loss += loss.item()

        # get losses
        epoch_loss = accum_loss / m
        net_loss += epoch_loss
        # apply scheduler after warmups
        warmup = epoch + 1 < warmups if scheduler else None
        if epoch + 1 > warmups and scheduler:
            scheduler.step(epoch_loss) 
        # check on checkpoint
        if checkpoint:
            saved = checkpoint.check(epoch_loss)
        # evaluate model
        if evaluator:
            curr_bleu = evaluator.evaluate(model)
            done = evaluator.done()
            model.train() # reset back
            bleu = evaluator.bleu

        # display info after end of epoch
        if verbose:
            train_printer(epoch_loss, epoch + 1, clock, curr_bleu, warmup, saved)
        # model meets bleu score
        if done:
            break

    net_loss /= epochs # avg accum loss over epochs
    # display info after end of training
    if verbose:
        train_printer(net_loss, None, clock, bleu, None, saved)
    return net_loss

def train_printer(loss, epoch=None, clock=None, bleu=None, warmup=None, saved=None):
    # basic info
    bar = f"{'-' * 70}"
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
    print(bar)
    print(info)

def predict(sequences, model, tokenizer, start, end, maxlen, special_tokens=False, device=None):
    # sequence inshape: (batch_size, src_len,)

    # setup
    tokenizer.inference()
    softmax = nn.Softmax(dim=-1)
    model.eval()

    # get prediction for encoded sequences
    token_ids = tokenizer.encode(sequences, model=True, module="encoder")
    predictions = []
    for ids in token_ids:

        # create src tensor
        ids = np.array(ids, dtype=int)
        src = torch.from_numpy(ids).unsqueeze(0).long() # (1, src_len)
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
            pred = torch.argmax(prob, dim=-1)[:, -1] # shape: (1, 1)
            pred = pred.contiguous().view(-1, 1)
            # add token to current tgt (1, output_size + 1)
            tgt = torch.cat((tgt, pred), dim=-1)

            # done prediction
            if pred.item() == end:
                break
        # store tokens of predictions
        predictions.append(tgt.squeeze().tolist())

    # create continuations with decoder
    predictions = tokenizer.decode(predictions, special_tokens=special_tokens, module="decoder")
    return predictions

def prompt(model, tokenizer, start, end, device=None):
    # setup
    model.eval()
    tokenizer.inference()
    softmax = nn.Softmax(dim=-1)

    # get input and encode
    sequence = input("Enter in the sequence of text:\n\n").strip()
    ids = tokenizer.encode(sequence, model=True, module="encoder")
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

        # get token with highest probability
        pred = torch.argmax(prob, dim=-1)[:, -1]
        pred = pred.contiguous().view(-1, 1)

        # combine prediction
        tgt = torch.cat((tgt, pred), dim=-1)

        # predicted end
        if pred.item() == end:
            break

    # maxlen exceeded
    return f"{sequence} -> {tokenizer.decode(tgt.tolist(), module='decoder')[0]}"

if __name__ == "__main__":
    pass

    

    
