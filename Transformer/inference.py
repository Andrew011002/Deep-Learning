import torch
import torch.nn as nn
import numpy as np

def predict(sequences, tokenizer, search, special_tokens=False):
    # inshape: sequences - (batch_size, seq_len*)

    # setup
    tokenizer.inference()
    tokenizer.truncon(search.maxlen)

    # get pred for encoded sequences
    token_ids = tokenizer.encode(sequences, model=True, module="encoder")
    predictions = []
    for ids in token_ids:
        # get prediction from ids
        sequence, score = search.search(ids)
        # store tokens of predictions
        predictions.append(sequence.squeeze().tolist())

    # translate token ids for predictions
    translations = tokenizer.decode(predictions, special_tokens=special_tokens, module="decoder")
    return translations

def prompt(tokenizer, search):

    # setup
    tokenizer.inference()
    tokenizer.truncon(search.maxlen)

    # get input & encode
    sequence = input("Enter in the sequence of text:\n\n").strip()
    ids = tokenizer.encode(sequence, model=True, module="encoder")
    sequence, score = search.search(ids)

    # translate token ids for prediction
    translation = tokenizer.decode(sequence.tolist(), module="decoder")[0]
    return translation

class DecoderSearch:
        
    def search(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

class Greedy(DecoderSearch):

    def __init__(self, model, start, end, maxlen, alpha=0.6, device=None) -> None:
        super().__init__()
        self.model = model
        self.start = start
        self.end = end
        self.maxlen = min(model.maxlen, maxlen)
        self.alpha = alpha
        self.device = device
        self.base = torch.tensor([1e-9]).unsqueeze(0)

    def search(self, ids):
        # inshape: ids - (ids_len, )
        model, start, end, maxlen, alpha, device, base = \
            self.model, self.start, self.end, self.maxlen, self.alpha, self.device, self.base
        
        # create src & tgt tensors src - (1, src_len) tgt - (1, 1)
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

    # auto-regress until end is predicted or output reaches maximum length
    while output[:, -1].item() != end and output.size(1) != maxlen:
        out = model(input, output, src_mask=mask)
        # shape: prob (1, vocab_size)
        prob = softmax(out)[:, -1]
        # shape: pred & logit - (1, 1) 
        pred = torch.argmax(prob, dim=-1)
        pred = pred.contiguous().view(-1, 1)
        logit = prob[:, pred.item()].unsqueeze(0)
        # combine token & logit shape: output & score - (1, output_len + 1)
        output = torch.cat((output, pred), dim=-1)
        score = torch.cat((score, logit), dim=-1)
    
    return output, log_score(score, alpha)
    
class Beam(DecoderSearch):

    def __init__(self, model, start, end, maxlen, beam_width=3, breadth=100, mode="best", alpha=0.6, device=None):
        super().__init__()
        self.model = model
        self.start = start
        self.end = end
        self.maxlen = min(model.maxlen, maxlen)
        self.beam_width = beam_width
        self.breadth = breadth
        self.alpha = alpha
        self.device = device
        self.base = torch.tensor([1e-9]).unsqueeze(0)
        if mode != "random" and mode != "best" and mode != "all":
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def search(self, ids):
        # inshape: ids - (ids_len, )
        model, start, end, maxlen, beam_width, breadth, mode, alpha, device, base = \
            self.model, self.start, self.end, self.maxlen, self.beam_width, \
            self.breadth, self.mode, self.alpha, self.device, self.base

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
        searches = sorted(searches, reverse=True, key=lambda cand: cand[1].item())[:beam_width]

        # return based on mode
        if mode == "random":
            i = np.random.choice(len(searches))
            candidate = searches[i]
            sequence, score = candidate
        if mode == "best":
            candidate = searches[0]
            sequence, score = candidate
        if mode == "all":
            sequence, score = [], []
            for b, s in searches:
                sequence.append(b)
                score.append(s)
        return sequence, score
                
def beam_search(model, input, candidate, end, maxlen, beam_width=3, searches=[], 
                max_breadth=100, alpha=0.6, mask=None):
    # inshape: input - (1, input_len)

    # setup
    model.eval()
    softmax = nn.Softmax(dim=-1)    

    # searches fully populated
    if len(searches) == max_breadth:
        return searches

    # shape: output & score - (1, output_len)
    output, score = candidate
    # base cases (predicted eos or up to maximum length)
    if output[:, -1].item() == end or output.size(1) == maxlen:
        # store beam & score
        searches.append((output, log_score(score, alpha)))
        return searches

    # get topk tokens within beam width from model output
    out = model(input, output, src_mask=mask)
    # shape: prob - (1, vocab_size)
    prob = softmax(out)[:, -1] 
    token = torch.argmax(prob, dim=-1)
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
    # inshape: tensor - (1, tensor_len)
    norm = 1 / np.power(tensor.size(1), alpha)
    log_prob = torch.log(tensor)
    return norm * torch.sum(log_prob, dim=-1, keepdim=True)
            
if __name__ == "__main__":
    pass

