from typing import Iterable
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

defaults = {"normalizer": normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase(), 
                            normalizers.StripAccents(), normalizers.NFD()]),
            "pretokenizer": pre_tokenizers.Whitespace(),
            "special tokens": {"unk": "[UNK]", "cls": "[CLS]", "sep": "[SEP]", 
                                "pad": "[PAD]", "mask": "[MASK]"}
            }

class WordPieceTokenizer:

    def __init__(self, vocab=None, unknown_token="[UNK]", prefix="##", normalizer=None, 
                    pre_tokenizer=None, special_tokens=None): 
        self.tokenizer = self.init_tokenizer(vocab, unknown_token, prefix, normalizer, 
                        pre_tokenizer, special_tokens)
        self.trained = False

    def init_tokenizer(self, vocab, unknown_token, prefix, normalizer, 
                        pre_tokenizer, special_tokens):
        # defaults
        if normalizer is None:
            normalizer = defaults["normalizer"]
        if pre_tokenizer is None:
            pre_tokenizer = defaults["pretokenizer"]
        if special_tokens is None:
            self.special_tokens = defaults["special tokens"]

        # set kwargs for retraining
        self.kwargs = {"unknown_token": unknown_token, "prefix": prefix, "normalizer": normalizer, 
                        "pre_tokenizer": pre_tokenizer, "special_tokens": special_tokens}
        
        # set tokenizer attrs
        tokenizer = Tokenizer(models.WordPiece(unk_token=unknown_token, vocab=vocab))
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoders.WordPiece(prefix)
        return tokenizer

    def train(self, size, corpus):
        # train tokenizer over corpus
        trainer = trainers.WordPieceTrainer(vocab_size=size, show_progress=False, 
                                special_tokens=list(self.special_tokens.values()))
        self.tokenizer.train_from_iterator(corpus, trainer=trainer)
        self.corpus = corpus
        self.size = size
        self.trained = True
        # init post processor for input sequences
        cls, sep = self.special_tokens["cls"], self.special_tokens["sep"]
        cls_id, sep_id = self.tokenizer.token_to_id(cls), self.tokenizer.token_to_id(sep)
        self.tokenizer.post_processor = processors.TemplateProcessing(
                                single=f"{cls}:0 $A:0 {sep}:0",
                                pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
                                special_tokens=[(cls, cls_id), (sep, sep_id)])

    def pruncate(self, maxlen, end=True):
        pad = self.special_tokens["pad"]
        pad_id = self.tokenizer.token_to_id(pad)
        direction = "right" if end else "left"
        # enable padding and truncation for maxlen
        self.tokenizer.enable_padding(direction=direction, pad_id=pad_id, 
                                    pad_token=pad, length=maxlen)
        self.tokenizer.enable_truncation(max_length=maxlen, direction=direction)

    def inference(self):
        # disable padding and truncation
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, data, model=False):
        # single sequence input
        if isinstance(data, str):
            tokens = self.tokenizer.encode(data)
            return tokens.ids if model else tokens.tokens
        # pair of sequence inputs
        if isinstance(data, tuple) and len(data) == 2:
            tokens = self.tokenizer.encode(*data)
            return tokens.ids if model else tokens.tokens
        encoded = []
        # list of sequences/sequence pairs
        for sequence in data:
            if isinstance(data, tuple):
                tokens = self.tokenizer.encode(*sequence)
            else:
                tokens = self.tokenizer.encode(sequence)
            # inputs for model or for reading
            if model:
                encoded.append(tokens.ids)
            else:
                encoded.append(tokens.tokens)
        return encoded

    def decode(self, data):
        # single encoded ids
        if isinstance(data[0], int):
            text = self.tokenizer.decode(data)
            return text
        decoded = []
        # list of encoded ids
        for encodings in data:
            text = self.tokenizer.decode(encodings)
            decoded.append(text)
        return decoded

    def vocab(self):
        return self.tokenizer.get_vocab()

    def save(self, filename):
        self.tokenizer.save(f"{filename}.json")

    def __len__(self):
        return self.size

def load_tokenizer(filename):
    return Tokenizer.from_file(f"{filename}.json")


if __name__ == "__main__":
    corpus = ["This is an escence to the beginning of what is Physics II. Phyics\
         II is a great course for learning how the things we work with every day\
             tend to work and operate. There's many special cases in which it is used"]
    tokenizer = WordPieceTokenizer()
    tokenizer.train(100, corpus)
    tokenizer.pruncate()
    print(tokenizer.encode("pad at all now its too long"))
    



    
    


    

    

   


