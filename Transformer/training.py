import torch
import torch.nn as nn
from utils import generate_masks

def train(model, optimizer, dataloader, epochs=5, device=None, verbose=True):

    if verbose:
        print("Training Started")
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
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
            src, tgt, out = src.long().to(device), tgt.long().to(device), out.long().to(device)
            # generate the mask
            src_mask, tgt_mask = generate_masks(src, tgt, model.pad_idx)
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

def predict(model, sequences, sos, maxlen, device=None):
    # src inshape: (batch_size, src_len,)

    # preprocess
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # create src tensor(s)
    src = torch.from_numpy(sequences).long().to(device) # (unknown, src_len)
    src = src.unsqueeze(0) if src.dim() < 2 else src # (batch_size, src_len)
    src_mask = (src != model.pad_idx).unsqueeze(-2).to(device)

    # create tgt tensor(s)
    tgt = torch.tensor([sos]).unsqueeze(0).long() # generate sos
    batch_size = src.size(0)
    tgt = tgt.repeat(batch_size, 1).to(device)

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
    
    return sequences, tgt.numpy()

def prompt(model, tokenizer, sos, eos, maxlen, device=None):
    # get input and tokenize
    text = [input("Enter in the sequence of text:\n\n").strip()]
    seq = tokenizer.encode(text, model=True)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # create src & tgt tensor
    src = torch.tensor(seq).unsqueeze(0).long().to(device)
    tgt = torch.tensor([sos]).unsqueeze(0).to(device)

    # predict sos from src until maxlen or eos token hit
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
        
        # predicted eos
        if pred.item() == eos:
            return tokenizer.deocde(tgt.squeeze().tolist(), special_tokens=False)
    # maxlen exceeded
    return tokenizer.decode(tgt.squeeze().tolist(), special_tokens=False)

if __name__ == "__main__":
    pass