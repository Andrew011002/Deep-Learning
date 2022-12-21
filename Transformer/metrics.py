import torch
import numpy as np
from training import predict
from collections import Counter

class Evaluator:

    def __init__(self, dataset, tokenizer, start, end, sample=32, ngrams=4, device=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.start = start
        self.end = end
        self.sample = sample
        self.ngrams = ngrams
        self.device = device

    def evaluate(self, model):
        # get inputs & references
        tokenizer, start, end, device = self.tokenizer, self.start, self.end, self.device
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        references = [pair[1] for pair in samples]
        maxlen = len(max(references, key=len))
        # generate prediction
        predictions = predict(inputs, model, tokenizer, start, end, maxlen, device)

        # get BLEU scroes
        for pred, ref in zip(predictions, references):
            # get ngram match score
            pred_counts, ref_counts = Counter(pred), Counter(ref)
            score = 0
            # find counts for each gram
            for gram in pred_counts:
                if gram in ref_counts:
                    # getting min yields the correct score
                    score += min(pred_counts[gram], ref[gram])
            # avg over len
            score /= len(pred)

if __name__ == "__main__":
    sequences = np.array([1])
    sequences = np.repeat(sequences, 32)
    print(sequences)



        