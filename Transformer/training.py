import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks, generate_nopeak_pad_mask, pad_tokens


def ascending_data(vocab_size, n, maxlen, sos, eos, pad_idx):
    inputs, outputs = [], []
    base = np.array([sos, eos])
    for i in range(n):
        seq_len = np.random.randint(1, maxlen + 1)
        token = np.random.randint(1, vocab_size) - maxlen
        token *= -1 if token < 0 else 1
        seq = np.arange(token, token + seq_len)
        seq = np.insert(base, 1, seq)
        src, tgt = seq, seq[::-1]
        inputs.append(src), outputs.append(tgt)

    inputs = pad_tokens(inputs, pad_idx=pad_idx, end=True)
    outputs = pad_tokens(outputs, pad_idx=pad_idx, end=True)
    return inputs, outputs



def train(model, optimizer, dataloader, epochs=5, device=None, verbose=False):

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

def predict(model, src, sos, maxlen, device=None):
    # src inshape: (batch_size, src_len,)

    # preprocess
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # encode src 
    src = torch.from_numpy(src).long().to(device) # (unknown, src_len)
    src = src.unsqueeze(0) if src.dim() < 2 else src # (batch_size, src_len)
    src_mask = (src != model.pad_idx).unsqueeze(-2).to(device)
    # embed and positionally encode src
    x = model.embeddings(src)
    x = model.pos_encoder(x)
    # pass through encoder layers
    e_out, attn = model.encoder(x, src_mask=src_mask)

    # create output tensor(s)
    output = torch.tensor([sos]).unsqueeze(0).long() # generate sos
    batch_size = src.size(0)
    output = output.repeat(batch_size, 1).to(device)

    # predict one token at a time
    while output.size(1) < maxlen:
        # decode from src and current output
        tgt_mask = generate_nopeak_pad_mask(output, model.pad_idx).to(device)
        # embed and positionally encoder output
        x = model.embeddings(output)
        x = model.pos_encoder(x)
        # pass through decoder layers
        d_out, attn1, attn2 = model.decoder(e_out, x, src_mask=src_mask, tgt_mask=tgt_mask)
        # unembed then apply softmax
        out = softmax(torch.matmul(d_out, model.wu.T))
        # get last token(s) of highest probability
        out = torch.argmax(out, dim=-1)[:, -1] # shape: (batch_size, 1)
        out = out.contiguous().view(-1, 1)
        # add token to current output (batch_size, output_size + 1)
        output = torch.cat((output, out), dim=-1)
    
    return output.numpy()

def prompt(model, tokenizer, sos, eos, maxlen, device=None):
    # get input and tokenize
    text = [input("Enter in the sequence of text:\n\n").strip()]
    seq = tokenizer.encode(text)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # embed and pos encode source
    src = torch.tensor(seq.astype(int)).unsqueeze(0).long().to(device)
    src = model.embeddings(src)
    src = model.pos_encoder(src)

    # pass through encoder
    e_out, attn = model.encoder(src, src_mask=None)

    # create output tensor
    out = torch.tensor([sos]).unsqueeze(0).to(device)
    # predict sos from src until maxlen or eos token hit
    while out.size(1) <= maxlen:  
        # embed and pos encode out
        x = model.embeddings(out)
        x = model.pos_encoder(x)

        # pass through decoder and unembded with probabilities
        d_out, attn, attn = model.decoder(e_out, x, src_mask=None, tgt_mask=None)
        prob = softmax(torch.matmul(d_out, model.wu.T))

        # get token with highest prediction
        pred = torch.argmax(prob, dim=-1)[:, -1]
        pred = pred.contiguous().view(-1, 1)

        # combine prediction
        out = torch.cat((out, pred), dim=-1)
        
        # predicted eos
        if pred.item() == eos:
            return tokenizer.deocde(out.numpy().squeeze())
    # maxlen exceeded
    return tokenizer.decode(out.numpy().squeeze())

if __name__ == "__main__":
    pass