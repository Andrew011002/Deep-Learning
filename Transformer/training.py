import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks

def train(dataloader, model, optimizer, scheduler=None, evaluator=None, 
    checkpoint=None, clock=None, epochs=1000, warmups=100, verbose=True, device=None):

    # setup
    if verbose:
        print("Training Started")
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    model.train()
    m = len(dataloader)
    net_loss = 0
    if clock:
        clock.start()

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

            # diplay info every fraction of an epoch
            if verbose:
                if (i + 1) % int(m * verbose) == 0 and (i + 1) != m:
                    print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")

        net_loss += accum_loss / m
        # apply scheduler after warmups
        if epoch + 1 > warmups and scheduler:
            scheduler.step(accum_loss / m) 
        # check on checkpoint
        if checkpoint:
            checkpoint.check(accum_loss / m)
        # display info after end of epoch
        if verbose:
            print(f"Epoch {epoch + 1} Complete | Epoch Loss: {accum_loss / m:.4f}")
        # evaluate mode
        if evaluator:
            evaluator.evaluate(model)
            # model meets bleu score (end training)
            if evaluator.done():
                break
            model.train() # reset back
        # show times 
        if clock:
            print(f"Epoch Duration: {clock.epoch()[3:]} | Elapsed Time: {clock.elapsed()}")

    net_loss /= epochs # avg accum loss over epochs
    # display info after end of training
    if verbose:
        print(f"Training Complete | Training Loss: {net_loss:.4f}")
    return net_loss

def retrain(dataloader, checkpoint, epochs=1000, warmups=100, verbose=True, device=None):
    
    # resume from checkpoint
    info = checkpoint.state_dict()
    model = info["model"]
    optimizer = info["optimizer"]
    scheduler = info["scheduler"]
    evaluator = info["evaluator"]
    epoch = info["epoch"]
    net_loss = info["loss"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    model.train()
    m = len(dataloader)

    if verbose:
        print("Training continued")

    for epoch in range(epoch, epochs):
        
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

            # diplay info every fraction of an epoch
            if verbose:
                if (i + 1) % int(m * verbose) == 0 and (i + 1) != m:
                    print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")

        net_loss += accum_loss / m
        # apply scheduler after warmups
        if epoch + 1 > warmups and scheduler:
            scheduler.step(accum_loss / m)     
        checkpoint.check(accum_loss / m)
        # display info after end of epoch
        if verbose:
            print(f"Epoch {epoch + 1} Complete | Epoch Loss: {accum_loss / m:.4f}")
        # evaluate model
        if evaluator:
            evaluator.evaluate(model)
            # model meets bleu score (end training)
            if evaluator.done():
                break
            model.train() # reset back

    net_loss /= epochs # avg accum loss over epochs
    # display info after end of training
    if verbose:
        print(f"Training Complete | Training Loss: {net_loss:.4f}")
    return net_loss

def predict(sequences, model, tokenizer, start, end, maxlen, special_tokens=False, device=None):
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

    # create continuations
    predictions = tokenizer.decode(predictions, special_tokens)
    return predictions

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
    pass
    

    
