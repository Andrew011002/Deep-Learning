import torch
import numpy as np
from training import predict
from collections import Counter

class Evaluator:

    def __init__(self, dataset, tokenizer, start, end, sample=32, ngrams=4, bleu=30, device=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.start = start
        self.end = end
        self.sample = sample
        self.ngrams = ngrams
        self.bleu = bleu
        self.device = device
        self.passed = True

    def evaluate(self, model):
        # get inputs & references
        tokenizer, start, end, ngrams, bleu, device = \
            self.tokenizer, self.start, self.end, self.ngrams,  self.bleu, self.device
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        references = [pair[1] for pair in samples]
        maxlen = len(max(references, key=len))
        # generate prediction
        predictions = predict(inputs, model, tokenizer, start, end, maxlen, device)

        # get BLEU scroes
        net_bleu = 0
        for pred, ref in zip(predictions, references):
            net_bleu += calc_ngrams_score(pred, ref, ngrams)["bleu"]
        # average the scores and indicate whether the model passes
        if net_bleu / len(inputs) >= bleu:
            self.passed = False

    def __call__(self):
        return self.passed

def calc_ngrams_score(prediction, reference, ngrams=4):
    metric = {"bleu": None, "percisions": []}
    # find score score for n grams
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
    metric["bleu"] = geometric_mean(scores)
    return metric
    
# gets the ngram for a sequence
def get_ngram(sequence, ngram):
    output = []
    for i in range(0, len(sequence), ngram):
        output.append(" ".join(sequence[i: i + ngram]))
    return output

# finds geometric mean over list of values
def geometric_mean(scores):
    return np.power(np.prod(scores), 1 / len(scores))

if __name__ == "__main__":
    prediction = "I have thirty six years too my guy".split()
    reference = "I have thirty six".split()
    print(calc_ngrams_score(prediction, reference))
    



        