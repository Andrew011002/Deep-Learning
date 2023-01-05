import torch
import torch.nn as nn
import numpy as np
from transformer import Transformer

def predict(sequences, model, tokenizer, start, end, maxlen, special_tokens=False, device=None):
    # inshape: sequences - (batch_size, seq_len)

    # setup
    tokenizer.inference()
    tokenizer.truncon(maxlen)
    softmax = nn.Softmax(dim=-1)
    model.eval()

    # get pred for encoded sequences
    token_ids = tokenizer.encode(sequences, model=True, module="encoder")
    predictions = []
    for ids in token_ids:
        # create src & tgt tensors shape: src - (1, src_len) tgt - (1, 1)
        ids = np.array(ids, dtype=int)
        src = torch.from_numpy(ids).unsqueeze(0).long() 
        tgt = torch.tensor([start]).unsqueeze(0).long() 
        # get mask for src pad
        src_mask = (src != model.pad_id).unsqueeze(-2) 
        # move tensors to device
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt = tgt.to(device)

        # predict one token at a time
        while tgt.size(1) < maxlen:
            # get model output & prob distribution
            out = model(src, tgt, src_mask=src_mask)
            prob = softmax(out) # shape: prob - (1, tgt_len, vocab_size)
            # get token with highest probability 
            pred = torch.argmax(prob, dim=-1)[:, -1] 
            pred = pred.contiguous().view(-1, 1) # shape: pred - (1, 1)
            # add token to current shape: tgt - (1, tgt_len + 1)
            tgt = torch.cat((tgt, pred), dim=-1)
            # done predicting
            if pred.item() == end:
                break
        
        # store tokens of predictions
        predictions.append(tgt.squeeze().tolist())

    # translate token ids for predictions
    translations = tokenizer.decode(predictions, special_tokens=special_tokens, module="decoder")
    return translations

def prompt(model, tokenizer, start, end, device=None):
    # setup
    model.eval()
    tokenizer.inference()
    softmax = nn.Softmax(dim=-1)

    # get input & encode
    sequence = input("Enter in the sequence of text:\n\n").strip()
    ids = tokenizer.encode(sequence, model=True, module="encoder")
    maxlen = len(ids[0])
    # create src & tgt tensor shape: src - (1, input_len) tgt - (1, 1)
    src = torch.tensor(ids).unsqueeze(0).long().to(device)
    tgt = torch.tensor([start]).unsqueeze(0).to(device)

    # get prediction for encoded sequence
    while tgt.size(1) <= maxlen:  
        # get model output & prob distribution
        out = model(src, tgt, src_mask=None, tgt_mask=None)
        prob = softmax(out)
        # get token with highest probability
        pred = torch.argmax(prob, dim=-1)[:, -1] 
        pred = pred.contiguous().view(-1, 1) # shape: pred - (1, 1)
        # add token to current shape: tgt - (1, tgt_len + 1)
        tgt = torch.cat((tgt, pred), dim=-1)
        # done predictiong
        if pred.item() == end:
            break

    # translate token ids for prediction
    translation = tokenizer.decode(tgt.tolist(), module="decoder")[0]
    return translation


def beam_search(model, beam, src, tgt, end, maxlen, src_mask):
    # inshape: src - (1, src_len) tgt - (1, 1)

    # setup
    softmax = nn.Softmax(dim=-1)
    searches = [[tgt, 0] for i in range(beam)]
    beams = []

    pass
            
            
if __name__ == "__main__":
    pass