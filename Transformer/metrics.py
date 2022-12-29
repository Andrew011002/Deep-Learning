import numpy as np
from training import predict
from collections import Counter

class Evaluator:

    def __init__(self, dataset, tokenizer, start, end, maxlen, 
        sample=32, ngrams=4, threshold=30, mode="geometric", device=None):
        if sample > len(dataset):
            raise ValueError(f"Sample size cannot exceed {len(dataset)}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.start = start
        self.end = end
        self.maxlen = maxlen
        self.sample = sample
        self.ngrams = ngrams
        self.threshold = threshold
        self.mode = mode
        self.device = device
        self.bleu = 0
        self.passed = True

    def evaluate(self, model):
        # get inputs & references
        tokenizer, start, end, maxlen, ngrams, mode, device, bleu = \
            self.tokenizer, self.start, self.end, self.maxlen, self.ngrams, \
            self.mode, self.device, self.bleu
        tokenizer.inference() # (doesn't pad or truncate)
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        references = [pair[1] for pair in samples]
        references = tokenizer.encode(references, model=False) # encode tokens
        # generate prediction
        predictions = predict(inputs, model, tokenizer, start, end, maxlen, 
            special_tokens=True, device=device)

        # get BLEU scroes
        net_bleu = 0
        for pred, ref in zip(predictions, references):
            net_bleu += calc_ngrams_score(pred, ref, mode, ngrams)["bleu"]
        # set best bleu score calculated
        self.bleu = max(net_bleu / self.sample, bleu)
        return self.bleu

    def done(self):
        return self.bleu >= self.threshold

def calc_ngrams_score(prediction, reference, mode="geometric", ngrams=4, start=None, end=None):
    # remove special tokens
    prediction = clean_sequence(prediction, start, end)
    reference = clean_sequence(reference, start, end)
    
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
        metric["percisions"].append(score / len(pred_ngram) * 100)
    
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
    for i in range(0, len(sequence)):
        seq_ngram = sequence[i: i + ngram]
        if len(seq_ngram) == ngram:
            output.append(" ".join(seq_ngram))
    return output

# removes start or end if contained within sequence
def clean_sequence(sequence, start, end):
    # sequence: str
    if sequence[0] == start:
        sequence = sequence[1:]
    if sequence[-1] == end:
        sequence - sequence[:-1]
    return sequence

# finds geometric mean over list of values
def geometric_mean(scores):
    n = len(scores)
    return np.power(np.prod(scores), 1 / n)

if __name__ == "__main__":
    prediction = "I have thirty six years old".split()
    reference = "I am thirty six years old".split()
    metric = calc_ngrams_score(prediction, reference, mode="geometric", ngrams=4)
    print(metric)



        