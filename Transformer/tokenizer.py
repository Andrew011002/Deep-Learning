from transformers import AutoTokenizer
from collections import defaultdict

class BytePairEncoder:

    def __init__(self, corpus, vocab_size, model=None, special_tokens=None):
        self.vocab_size = vocab_size
        self.corpus = corpus
        self.model = model
        self.frequencies = self.create_frequencies(corpus, model)
        self.vocab = self.create_vocab(self.frequencies, special_tokens)
        self.splits = self.generate_splits(self.frequencies)
        self.merges = defaultdict(str)

    def create_frequencies(self, corpus, model):
        
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
    
    def create_vocab(self, frequencies, special_tokens):
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
        if pair_frequencies:
            max_pair = max(pair_frequencies, key=pair_frequencies.get)
            return max_pair
        else:
            print(f"Max vocab size of {len(self.vocab)} reached")

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
        
        # update until vocab size reached
        if len(self.vocab) == self.vocab_size:
            return None

        # find pair with highest frequency
        pair = self.compute_max_pair(self.splits, self.frequencies)

        # update if new pair exists
        if pair:
            # update merges
            self.merges[pair] = pair[0] + pair[1]
            # update splits
            self.splits = self.merge_pair(pair, self.splits, self.frequencies)
            # update vocab
            self.vocab.append(pair[0] + pair[1])
            return self.update()

    def train(self):
        # find new merges until desired vocab size    
        self.update()

    def tokenize(self, data):
        # get tokenizer for pre-tokenization
        if self.model is None:
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
                # try to find a merge in the split
                for index, split in enumerate(splits):
                    i = 0
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

    def get_vocab(self):
        return self.vocab

    def get_merges(self):
        return self.merges

    def get_frequencies(self):
        return self.frequencies


if __name__ == "__main__":
    corpus = ["hello my name is doug", "doug is a dog who is a good boy", 
            "this is just a small example of testing the encoder", 
            "just another piece of information where we can byte-pair encode",
            "another day, another dollar you know what they say"]

    bpe = BytePairEncoder(corpus, 100)
    bpe.train()
    sentence = ["what's up doug! You seem like a pretty cool dog"]
    print(bpe.get_vocab())
    tokenized = bpe.tokenize(sentence)
    print(tokenized)


