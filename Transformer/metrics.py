import numpy as np
from training import predict
from collections import Counter

class Evaluator:

    def __init__(self, dataset, tokenizer, sos, eos, maxlen, 
        sample=32, ngrams=4, threshold=30, mode="geometric", device=None):
        if sample > len(dataset):
            raise ValueError(f"Sample size cannot exceed {len(dataset)}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sos = sos
        self.eos = eos
        self.maxlen = maxlen
        self.sample = sample
        self.ngrams = ngrams
        self.threshold = threshold
        self.mode = mode
        self.device = device
        self.bleu = 0
        self.passed = True

    def evaluate(self, model):
        tokenizer, sos, eos, maxlen, ngrams, mode, device, bleu = \
            self.tokenizer, self.sos, self.eos, self.maxlen, self.ngrams, \
            self.mode, self.device, self.bleu
        # (disable padding)
        tokenizer.inference() 
        tokenizer.truncon(maxlen, end=True)
        
        # get inputs & references
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        references = [pair[1] for pair in samples]
        references = tokenizer.encode(references, model=False, module="decoder") # encode tokens
        # generate prediction
        predictions = predict(inputs, model, tokenizer, tokenizer[sos], tokenizer[eos], maxlen, 
            special_tokens=True, device=device)

        # get BLEU scroes
        net_bleu = 0
        for pred, ref in zip(predictions, references):
            pred = pred.split()
            net_bleu += calc_ngrams_score(pred, ref, mode, ngrams, sos, eos)["bleu"]
        # set best bleu score calculated
        self.bleu = max(net_bleu / self.sample, bleu)
        return net_bleu / self.sample

    def done(self):
        return self.bleu >= self.threshold

def calc_ngrams_score(prediction, reference, mode="geometric", ngrams=4, sos=None, eos=None):
    # remove special tokens
    prediction = clean_sequence(prediction, sos, eos)
    reference = clean_sequence(reference, sos, eos)
    
    # find score score for n grams
    metric = {"bleu": None, "percisions": [], \
        "prediction length": len(prediction), "reference length": len(reference)}
    for n in range(1, ngrams + 1):
        score = 0
        # generate ngrams
        pred_ngram, ref_ngram = get_ngram(prediction, n), \
            get_ngram(reference, n)

        # get counts for grams in ngrams
        pred_counts, ref_counts = Counter(pred_ngram), Counter(ref_ngram)
        # find matching ngrams (choose min to find correct matches for ngram)
        for ngram in pred_counts:
            if ngram in ref_counts:
                score += min(pred_counts[ngram], ref_counts[ngram])
        # avg score over length of prediction and accumulate score
        if len(pred_ngram) > 0:
            metric["percisions"].append(score / len(pred_ngram) * 100)
        else:
            metric["percisions"].append(0)

    # calculate bleu score
    scores = metric["percisions"]
    # set bleu based on type of mean
    if mode == "geometric":
        metric["bleu"] = geometric_mean(scores)
    elif mode == "standard":
        metric["bleu"] = np.mean(scores)
    # invalid mean
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return metric
    
# gets the ngram for a sequence
def get_ngram(sequence, ngram):
    output = []
    for i in range(len(sequence)):
        seq_ngram = sequence[i: i + ngram]
        if len(seq_ngram) == ngram:
            output.append(" ".join(seq_ngram))
    return output

# removes sos or eos if contained within sequence
def clean_sequence(sequence, sos, eos):
    # sequence: str
    if sequence[0] == sos:
        sequence = sequence[1:]
    if sequence[-1] == eos:
        sequence = sequence[:-1]
    return sequence

# finds geometric mean over list of values
def geometric_mean(scores):
    n = len(scores)
    return np.power(np.prod(scores), 1 / n)

if __name__ == "__main__":
    pass



        