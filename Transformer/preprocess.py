import numpy as np
import re
import unidecode
import nltk
import spacy
from word2number import w2n
from nltk.corpus import stopwords as STOPWORDS, wordnet
from nltk.stem import WordNetLemmatizer
from torchtext.datasets import AG_NEWS
from transformers import AutoTokenizer
from string import punctuation as PUNCTUATION, digits as DIGITS

nlp = spacy.load("en_core_web_lg")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


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

    def preprocess(self, data, operations=None):

        if operations is None:
            operations = {"accent": dict(), "html": dict(pattern=None), "punctuation": dict(punctuation=None),
                        "numbers": dict(), "stopwords": dict(stopwords=None), "lemmatize": dict(lemmatizer=None), 
                        "whitespaces": dict(),}

        mappings = {"html": self.remove_html, "punctuation": self.remove_punctuation, "digits": self.remove_digits, 
                    "stopwords": self.remove_stopwords, "whitespaces": self.remove_whitespaces, "accents": self.replace_accents,
                    "numbers": self.replace_numbers, "lemmatize": self.lemmatize}

        for name, kwargs in operations.items():
            operation = mappings[name]
            operation(data, **kwargs)


if __name__ == '__main__':

    # BASIC TOKENIZER
    # tokenizer = Tokenizer()            
    # sentence = "Hello, my name is Andrew and this is an example sentence. I'm not to sure how the tokenizer works so this is an attempt!"
    # ids = tokenizer.encode_(sentence)
    # print(ids)
    # original = tokenizer.decode_(ids)
    # print(original)

    # ids = tokenizer.encode_(sentence, model=True)
    # print(ids)
    # original = tokenizer.decode_(ids)
    # print(original)


    # BASIC PREPROCESSING

    data = [text for label, text in list(iter(AG_NEWS('./data', split='test')))]
    process = Process()
    print("(Original)", data[79])
    process.remove_punctuation(data)
    print("(Remove Punctuation)", data[79])
    process.remove_words(data, ["technology", "companies"])
    print("(Remove Specific Words)", data[79])
    # process.remove_digits(data)
    process.replace_numbers(data)
    print("(Replace Numbers)", data[79])
    process.remove_stopwords(data)
    print("(Remove Stopwords)", data[79])
    process.lemmatize(data, lemmatizer=nltk_lemmatizer)
    print("(Lemmatize Text)", data[79])

    # TOKENIZING TEXT COMPLETELY

    tokens = Tokenizer().encode(data, model=True)
    print("(Tokens)", tokens[79])
    decoded = Tokenizer().decode(tokens)
    print("(Token Values)", decoded[79])
