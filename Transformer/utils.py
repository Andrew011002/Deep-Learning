from string import punctuation
import numpy as np
from collections import Counter

def clean(data, punctuation=None, ignore=None):

    for i, sequence in enumerate(data):
        sequence = sequence.split()
        for j, word in enumerate(sequence):
            word = ''.join(c if c not in punctuation else ' ' for c in word)
            sequence[j] = word
        
        sequence = ' '.join(w for w in sequence if w not in ignore)
        data[i] = sequence

    return data

            