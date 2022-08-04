import numpy as np
import re
from torchtext.datasets import AG_NEWS
from transformers import AutoTokenizer
from string import punctuation as PUNCTUATION, digits as DIGITS
from nltk.corpus import stopwords as STOPWORDS

class Tokenizer:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def encode(self, text, model=False):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if model:
            ids = self.tokenizer.prepare_for_model(ids)["input_ids"]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)






class Process:

    def __init__(self) -> None:
        pass

    def remove_punctuation(self, data, punctuation=None, lower=True):

        if punctuation is None:
            punctuation = PUNCTUATION

        table = str.maketrans('', '', punctuation)
        for i, text in enumerate(data):
            text = text.lower() if lower else text
            data[i] = text.translate(table)

    def remove_digits(self, data, digits=None, lower=True):

        if digits is None:
            digits = DIGITS

        table = str.maketrans('', '', digits)
        for i, text in enumerate(data):
            text = text.lower() if lower else text
            data[i] = text.translate(table)

    def remove_stopwords(self, data, stopwords=None, lower=True):

        if stopwords is None:
            stopwords = STOPWORDS.words("english")

        for i, text in enumerate(data):
            text = text.lower() if lower else text
            text = text.split()
            text = " ".join(w for w in text if w not in stopwords)
            data[i] = text

    def remove_html(self, data, pattern=None, lower=True):

        if pattern is None:
            pattern = "<.*?>"

        cleaner = re.compile(pattern)
        for i, text in enumerate(data):
            text = text.lower() if lower else text
            data[i] = re.sub(cleaner, '', text)

    def remove_words(self, data, words=None, substrings=False, lower=True):

        if words is None:
            words = [""]

        if substrings:
            cleaner = re.compile("|".join(map(re.escape, words)))
        else:
            cleaner = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, words)))

        for i, text in enumerate(data):
            text = text.lower() if lower else text
            data[i] = re.sub(cleaner, '', text)

    def preprocess(self, data, operations=None):

        if operations is None:
            operations = {"html": dict(pattern=None), "punctuation": dict(punctuation=None),
                        "digits": dict(digits=None), "stopwords": dict(stopwords=None)}

        mappings = {"html": self.remove_html, "punctuation": self.remove_punctuation,
                    "digits": self.remove_digits, "stopwords": self.remove_stopwords}
        for name, kwargs in operations.items():
            operation = mappings[name]
            operation(data, **kwargs)


if __name__ == '__main__':
    # Tokenizer example

    # tokenizer = Tokenizer()            
    # sentence = "Hello, my name is Andrew and this is an example sentence. I'm not to sure how the tokenizer works so this is an attempt!"
    # ids = tokenizer.encode(sentence)
    # print(ids)
    # original = tokenizer.decode(ids)
    # print(original)

    # ids = tokenizer.encode(sentence, model=True)
    # print(ids)
    # original = tokenizer.decode(ids)
    # print(original)



    data = [text for label, text in list(iter(AG_NEWS('./data', split='test')))]
    process = Process()
    print(1, data[10])
    process.remove_punctuation(data)
    print(2, data[10])
    process.remove_words(data, ["technology", "companies"])
    print(3, data[10])
    process.remove_digits(data)
    print(4, data[10])
    process.remove_stopwords(data)
    print(5, data[10])



