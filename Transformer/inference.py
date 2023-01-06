import torch
import torch.nn as nn
import numpy as np
from pytorch_beam_search.seq2seq import beam_search, greedy_search
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
    src = torch.tensor(ids).unsqueeze(0).long()
    tgt = torch.tensor([start]).unsqueeze(0)
    # move to device
    src = src.to(device)
    tgt = tgt.to(device)

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

class Greedy:

    def __init__(self, model, start, end, maxlen, alpha=0.6) -> None:
        self.model = model
        self.start = start
        self.end = end
        self.maxlen = maxlen
        self.alpha = alpha
        self.base = torch.tensor([1e-9]).unsqueeze(0)

    def search(self, ids, device=None):
        # inshape: ids = (ids_len,)

        model, start, end, maxlen, alpha, base = \
            self.model, self.start, self.end, self.maxlen, self.alpha, self.base
        
        # create src & tgt tensors
        ids = np.array(ids, dtype=int)
        src = torch.from_numpy(ids).unsqueeze(0).long()
        tgt = torch.tensor([start]).unsqueeze(0).long()
        # generate mask
        mask = (src != model.pad_id).unsqueeze(-2)
        # move to device
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)

        # predict best token given source
        sequence, score = greedy_search(model, src, (tgt, base), end, maxlen, 
                                        alpha=alpha, mask=mask)
        return sequence, score

def greedy_search(model, input, candidate, end, maxlen, alpha=0.6, mask=None):
    # inshape: input - (1, input_len) 

    # setup
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # shape: output & score - (1, 1)
    output, score = candidate

    # auto-regress until eos is predicted or output reaches maximum length
    while output[:, -1] != end and output.size(1) != maxlen:
        out = model(input, output, src_mask=mask)
        # shape: prob (batch_size, 1, vocab_size)
        prob = softmax(out)
        # shape: pred - (1, 1)
        pred = torch.argmax(prob, dim=-1)[:, -1]
        pred = pred.contiguous().view(-1, 1)
        score = torch.cat((score, prob[:, pred.item()]), dim=-1)
        # combine with output
        output = torch.cat((output, pred), dim=-1)
    
    return [output], [log_score(score, alpha)]
    
class Beam:

    def __init__(self, model, start, end, maxlen, beam_width=3, breadth=100, alpha=0.6):
        self.model = model
        self.start = start
        self.end = end
        self.maxlen = maxlen
        self.beam_width = beam_width
        self.breadth = breadth
        self.aplpha = alpha
        self.base = torch.tensor([1e-9]).unsqueeze(0)

    def search(self, ids, device=None):
        # inshape: ids - (ids_len)
        model, start, end, maxlen, beam_width, breadth, alpha, base = \
            self.model, self.start, self.end, self.maxlen, self.beam_width, \
            self.breadth, self.aplpha, self.base

        # create src & tgt tensors shape: src - (1, src_len) tgt - (1, 1)
        ids = np.array(ids, dtype=int)
        src = torch.tensor(ids).unsqueeze(0).long()
        tgt = torch.tensor([start]).unsqueeze(0).long()
        # generate mask
        mask = (src != model.pad_id).unsqueeze(-2)
        # move to device
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)

        # search beams until populated
        searches = beam_search(model, src, (tgt, base), end, maxlen, beam_width, 
                                max_breadth=breadth, alpha=alpha, mask=mask)
        # get topk beams & scores
        beams, scores = [], []
        searches = sorted(searches, reverse=True, key=lambda cand: cand[1].item())[:beam_width]
        # return as pair
        for beam, score in searches:
            beams.append(beam)
            scores.append(score)
        return beams, scores
        
def beam_search(model, input, candidate, end, maxlen, beam_width=3, searches=[], 
                max_breadth=100, alpha=0.6, mask=None):
    # inshape: input - (1, input_len)

    # searches fully populated
    if len(searches) == max_breadth:
        return searches

    # shape: output & score - (1, output_len)
    output, score = candidate
    # base cases (predicted eos or up to maximum length)
    if output[:, -1] == end or output.size(1) == maxlen:
        # store beam & score
        searches.append((output, log_score(score, alpha)))
        return searches

    # setup
    softmax = nn.Softmax(dim=-1)    
    model.eval()

    # get topk tokens within beam width from model output
    out = model(input, output, src_mask=mask)
    # shape: prob - (1, vocab_size)
    prob = softmax(out)[:, -1] 
    # shape: pred & indices - (1, beam_width)
    pred, indices = torch.topk(prob, k=beam_width, dim=-1)

    # expand search for possible tokens within beam width
    for i in range(pred.size(-1)):
        token, logit = indices[:, i].unsqueeze(-1), pred[:, i].unsqueeze(-1)
        # combine token & logit shape: beam & beam_score - (1, output_len + 1)
        beam, beam_score = torch.cat((output, token), dim=-1), torch.cat((score, logit), dim=-1)
        # continue search with new candidate
        candidate = (beam, beam_score)
        searches = beam_search(model, input, candidate, end, maxlen, beam_width, 
                            max_breadth=max_breadth, alpha=alpha, mask=mask)

    # return all searches
    return searches

def log_score(tensor, alpha=0.6):
    norm = 1 / np.power(tensor.size(1), alpha)
    log_prob = torch.log(tensor)
    return norm * torch.sum(log_prob, dim=-1, keepdim=True)
            
if __name__ == "__main__":
    beam_width = 3
    maxlen = 10
    start, end, pad = 1, 2, 0
    model = Transformer(vocab_enc=100, vocab_dec=100, maxlen=maxlen, pad_id=pad)
    ids = [1] + [np.random.randint(0, 100) for i in range(maxlen - 2)] + [2]
    beam = Beam(model, start, end, maxlen, beam_width, breadth=25, alpha=0.6)
    beams, scores = beam.search(ids)
    print(beams)
    print()
    print(scores)
