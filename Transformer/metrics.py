import torch
import numpy as np

class Evaluator:

    def __init__(self, dataset, tokenizer, sample=32, ngrams=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sample = sample
        self.ngrams = ngrams

    def evaluate(self, model):
        # inference mode
        model.eval()
        self.tokenizer.inference()
        softmax = torch.nn.Softmax(dim=-1)

        # get inputs and labels
        samples = self.dataset.sample(self.sample)
        inputs = [pair[0] for pair in samples]
        labels = [pair[1] for pair in samples]

        # convert to tokens
        


        