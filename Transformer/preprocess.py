import numpy as np
import re
import unidecode
import nltk
import spacy
from collections import defaultdict
from word2number import w2n
from nltk.corpus import stopwords as STOPWORDS, wordnet
from nltk.stem import WordNetLemmatizer
from torchtext.datasets import AG_NEWS
from transformers import AutoTokenizer
from string import punctuation as PUNCTUATION, digits as DIGITS

# nlp = spacy.load("en_core_web_lg")
nlp = None
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].lower()
    mappings = dict(a=wordnet.ADJ, n=wordnet.NOUN, v=wordnet.VERB, 
                    r=wordnet.ADV, j=wordnet.ADJ, s=wordnet.ADJ_SAT)
    return mappings.get(tag, wordnet.NOUN)

def spacy_lemmatizer(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


nltk_wnl = WordNetLemmatizer()
def nltk_lemmatizer(text):
    text = text.split()
    return [nltk_wnl.lemmatize(w, get_pos(w)) for w in text]


class Tokenizer:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def encode_(self, text, model=False):

        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if model:
            ids = self.tokenizer.prepare_for_model(ids)["input_ids"]
        return ids

    def decode_(self, ids):
        return self.tokenizer.decode(ids)

    def encode(self, data, model=False):

        encoded = []
        for text in data:
            encoded.append(self.encode_(text, model))
        return np.array(encoded, dtype=object)

    def decode(self, tokens):

        decoded = []
        for ids in tokens:
            decoded.append(self.decode_(ids))
        return np.array(decoded, dtype=object)        

class BytePairEncoder:

    def __init__(self, corpus, vocab_size, model=None, special_tokens=None):
        self.vocab_size = vocab_size
        self.corpus = corpus
        self.model = model
        self.frequencies = self.get_word_freqs(corpus, model)
        self.vocab = self.get_characters(self.frequencies, special_tokens)
        self.splits = self.generate_splits(self.frequencies)
        self.merges = defaultdict(str)

    def get_word_freqs(self, corpus, model):
        
        # get tokenizer for pre-tokenization
        if model is None:
            model = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model)

        frequencies = defaultdict(int)
        for text in corpus:
            # get words and their offsets
            words = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            # update word frequency
            for word, offset in words:
                frequencies[word] += 1

        return frequencies
    
    def get_characters(self, frequencies, special_tokens):
        characters = set()
        # add any special tokens
        if special_tokens:
            for token in special_tokens:
                characters.add(token)

        # get unique set of all characters in word corpus
        for word in frequencies.keys():
            for char in word:
                # only adds if the character is not in the set
                characters.add(char)

        return sorted(list(characters))

    def generate_splits(self, frequencies):
        splits = dict()
        # create word character break-up pairings
        for word in frequencies.keys():
            splits[word] = [char for char in word]
        return splits

    def compute_max_pair(self, splits, frequencies):
        pair_frequencies = defaultdict(int)
        # get the split for all words in the corpus
        for word, freq in frequencies.items():
            split = splits[word]
            n = len(split)
            # only find pair frequecies for splits larger than 1
            if n > 1:
                for i in range(n - 1):
                    # update pair frequency
                    pair = (split[i],  split[i + 1])
                    pair_frequencies[pair] += freq

        # get most common pair
        max_pair = max(pair_frequencies, key=pair_frequencies.get)
        return max_pair

    def merge_pair(self, pair, splits, frequencies):
        
        # get the split for words
        for word in frequencies.keys():
            split = splits[word]
            # only merge for words with splits larger than 1
            if len(split) == 1:
                continue
            i = 0
            # find the pair in the word if it exists
            while i < len(split) - 1:
                # exists in words
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    # merge
                    split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
                # not found look further
                else:
                    i += 1
            # update the split
            splits[word] = split

        return splits 

    def update(self):
        # find pair with highest frequency
        pair = self.compute_max_pair(self.splits, self.frequencies)
        # update merges
        self.merges[pair] = pair[0] + pair[1]
        # update splits
        self.splits = self.merge_pair(pair, self.splits, self.frequencies)
        # update vocab
        self.vocab.append(pair[0] + pair[1])

    def train(self):
        # find new merges until desired vocab size
        while len(self.vocab) < self.vocab_size:
            self.update()

    def tokenize(self, data, model=None):
        # get tokenizer for pre-tokenization
        if model is None:
            model = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model)

        tokenized = []
        for text in data:
            # get words and their offsets
            words = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            # get the splits for text
            splits = [[char for char in word] for word, offset in words]
            # modify word split if it contains a merge
            for pair, merge in self.merges.items():
                i = 0
                # try to find a merge in the split
                for index, split in enumerate(splits):
                    # only merge for words with splits larger than 1
                    if len(split) == 1:
                        continue
                    # look for merge within split
                    while i < len(split) - 1:
                        # merge within split
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2:]
                        # merge not in split keep looking
                        else:
                            i += 1
                    # update split based on what was modified
                    splits[index] = split
            # create the modifed text with merges and add to data
            tokens = sum(splits, [])
            tokenized.append(tokens)
        return tokenized
    
class PreProcess:

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


    def remove_whitespaces(self, data):

        for i, text in enumerate(data):
            data[i] = re.sub(" +", " ", text)

    def replace_accents(self, data, lower=True):
        
        for i, text in enumerate(data):
            text = text.lower() if lower else text
            data[i] = unidecode.unidecode(text)

    def replace_numbers(self, data, lower=True):

        for i, text in enumerate(data):
            text = text.lower() if lower else text
            doc = nlp(text)
            tokens = []
            for token in doc:
                if token.pos_ == "NUM":
                    try:
                        tokens.append(str(w2n.word_to_num(token.text)))
                    except:
                        tokens.append(token.text)
                else:
                    tokens.append(token.text)
            data[i] = " ".join(tokens)

    def lemmatize(self, data, lemmatizer=None, lower=True):

        if lemmatizer is None:
            lemmatizer = spacy_lemmatizer
        
        for i, text in enumerate(data):
            text = text.lower() if lower else text
            text = lemmatizer(text)
            data[i] = " ".join(text)

    def normalize(self, data, model=None):

        if model is None:
            model = "bert-base-uncased"

        tokenizer = AutoTokenizer.from_pretrained(model)

        for i, text in enumerate(data):
            text = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
            data[i] = text

    def preprocess(self, data, operations=None):

        if operations is None:
            operations = {"accent": dict(), "html": dict(pattern=None), "punctuation": dict(punctuation=None),
                            "numbers": dict(), "whitespaces": dict(),}

        mappings = {"html": self.remove_html, "punctuation": self.remove_punctuation, "digits": self.remove_digits, 
                    "stopwords": self.remove_stopwords, "whitespaces": self.remove_whitespaces, "accents": self.replace_accents,
                    "numbers": self.replace_numbers, "lemmatize": self.lemmatize}

        for name, kwargs in operations.items():
            operation = mappings[name]
            operation(data, **kwargs)


if __name__ == '__main__':
    pass

    
    
    
    