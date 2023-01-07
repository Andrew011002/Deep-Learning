import numpy as np
from inference import predict
from collections import Counter

class Evaluator:

    def __init__(self, dataset, tokenizer, search, sample=32, ngrams=4, bleu_goal=30, mode="geometric"):
        if sample > len(dataset):
            raise ValueError(f"Sample size cannot exceed {len(dataset)}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.search = search
        self.sample = sample
        self.ngrams = ngrams
        self.bleu_goal = bleu_goal
        self.mode = mode
        self.bleu = 0
        self.passed = True

    def evaluate(self):
        tokenizer, search, ngrams, mode = \
            self.tokenizer, self.search, self.ngrams, self.mode
        
        # get inputs & references
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        references = [pair[1] for pair in samples]
        references = tokenizer.encode(references, model=False, module="decoder") # encode tokens
        # generate predictions
        predictions = predict(inputs, tokenizer, search, special_tokens=True)

        # get BLEU scroes between predictions and references
        sos, eos = tokenizer[search.start], tokenizer[search.end]
        net_bleu = 0
        for pred, ref in zip(predictions, references):
            pred = pred.split()
            net_bleu += calc_ngrams_score(pred, ref, mode, ngrams, sos, eos)["bleu"]

        # set best bleu score calculated
        net_bleu /= len(predictions)
        self.bleu = max(net_bleu, self.bleu)
        return net_bleu

    def done(self):
        return self.bleu >= self.bleu_goal

def calc_ngrams_score(prediction, reference, mode="geometric", ngrams=4, sos=None, eos=None):
    # inshape: prediction - (prediction_len, ) reference - (reference_len, )

    # setup
    metric = {"bleu": None, "percisions": [], "prediction length": len(prediction), 
                "reference length": len(reference)}
    # remove special tokens
    prediction = clean_sequence(prediction, sos, eos)
    reference = clean_sequence(reference, sos, eos)
    
    # find scores for each ngram
    for n in range(1, ngrams + 1):
        score = 0 # reset every ngram

        # generate ngrams & gram counts for pred & ref
        pred_ngram, ref_ngram = get_ngram(prediction, n), \
            get_ngram(reference, n)
        pred_counts, ref_counts = Counter(pred_ngram), Counter(ref_ngram)

        # find matching ngrams (choose min to find correct matches for ngram)
        for ngram in pred_counts:
            if ngram in ref_counts:
                score += min(pred_counts[ngram], ref_counts[ngram])
        # avg score over length of pred ngram & append to scores
        if len(pred_ngram) > 0:
            metric["percisions"].append(score / len(pred_ngram) * 100)
        # pred ngram was empty (predicted eos after sos)
        else:
            metric["percisions"].append(0)

    # calc bleu score based on mode
    scores = metric["percisions"]
    if mode == "geometric":
        metric["bleu"] = geometric_mean(scores)
    elif mode == "standard":
        metric["bleu"] = np.mean(scores)
    # invalid mode
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return metric
    
def get_ngram(sequence, ngram):
    # get ngram for sequence
    output = []
    for i in range(len(sequence)):
        seq_ngram = sequence[i: i + ngram]
        # valid ngram
        if len(seq_ngram) == ngram:
            output.append(" ".join(seq_ngram))
    return output

def clean_sequence(sequence, sos, eos):
    if sequence[0] == sos:
        sequence = sequence[1:]
    if sequence[-1] == eos:
        sequence = sequence[:-1]
    return sequence

def geometric_mean(scores):
    n = len(scores)
    return np.power(np.prod(scores), 1 / n)

if __name__ == "__main__":
    pass



        