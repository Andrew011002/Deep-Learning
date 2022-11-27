
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

class BaseTokenizer:

    def __init__(self, unknown_token=None, model=None, normalizer=None, pre_tokenizer=None, 
                decoder=None, trainer=None, special_tokens=None):
        self.tokenizer = self.init_tokenizer(unknown_token, model, normalizer,
            pre_tokenizer, decoder, trainer, special_tokens)
        self.trained = False

    def init_tokenizer(self, unknown_token, model, normalizer, 
        pre_tokenizer, deocder, trainer, special_tokens):

        # set tokenizer attrs
        tokenizer = Tokenizer(model(unk_token=unknown_token))
        if normalizer is not None:
            tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = deocder
        self.trainer = trainer
        self.special_tokens = special_tokens
        return tokenizer

    def train(self, corpus, size=None):
        # default vocab size for wpt
        if size is None:
            size = 30000

        # train tokenizer over corpus
        trainer = self.trainer(vocab_size=size, show_progress=False, 
                    special_tokens=list(self.special_tokens.values()))
        self.corpus = corpus
        self.tokenizer.train_from_iterator(corpus, trainer=trainer)
        # init post processor for input sequences
        start, end = self.special_tokens["start"], self.special_tokens["end"]
        start_id, end_id = self.tokenizer.token_to_id(start), self.tokenizer.token_to_id(end)
        self.tokenizer.post_processor = processors.TemplateProcessing(
                                single=f"{start}:0 $A:0 {end}:0",
                                pair=f"{start}:0 $A:0 {end}:0 $B:1 {end}:1",
                                special_tokens=[(start, start_id), (end, end_id)])
        
        # set vocab size and trained
        self.size = len(self.vocab())
        self.trained = True

    # enables padding and truncation
    def pruncate(self, maxlen=None, end=True):
        # only enable when trained
        if self.trained:
            # get args
            pad = self.special_tokens["pad"]
            pad_id = self.tokenizer.token_to_id(pad)
            direction = "right" if end else "left"
            # enable padding
            self.tokenizer.enable_padding(direction=direction, pad_id=pad_id, 
                                        pad_token=pad, length=maxlen)
            # enable truncation (if possible)
            if maxlen is not None:
                self.tokenizer.enable_truncation(max_length=maxlen, direction=direction)

    # disable padding and truncation
    def inference(self):
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    # encodes input to tokens or ids
    def encode(self, data, model=False):
        # handle single sequence or pair
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, tuple) and len(data) == 2:
            data = [data]

        # get encodings
        encodings = self.tokenizer.encode_batch(data)
        encoded = []
        for encoding in encodings:
            # add tokens or ids
            if model:
                encoded.append(encoding.ids)
            else:
                encoded.append(encoding.tokens)
        return encoded
        
    # decodies ids to tokens
    def decode(self, data, special_tokens=True):
        # decode the sequence(s)
        tokens = self.tokenizer.decode_batch(data, skip_special_tokens=not special_tokens)
        return tokens

    # returns entire vocab
    def vocab(self):
        return self.tokenizer.get_vocab()

    # saves tokenizer as json
    def save(self, filename):
        self.tokenizer.save(f"{filename}.json")

    def add(self, tokens):
        self.tokenizer.add_tokens(tokens)

    # total vocabulary
    def __len__(self):
        return self.size

    # gets id of token in vocab
    def __getitem__(self, token):
        vocab = self.vocab()
        return vocab.get(token, KeyError(f"{token} not in vocab"))

    # trys to encode then decode if possible
    def __call__(self, data, model=False, special_tokens=True):
        try: return self.encode(data, model=model)
        except: 
            try: return self.decode(data, special_tokens=special_tokens)
            except:
                raise ValueError(f"Can't interpret input {data}")

class WordPieceTokenizer(BaseTokenizer):

    def __init__(self, unknown_token="[UNK]"): 
        super().__init__(unknown_token, model=models.WordPiece, 
        normalizer=normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase(), 
                    normalizers.StripAccents(), normalizers.NFD()]), 
        pre_tokenizer=pre_tokenizers.Whitespace(), decoder=decoders.WordPiece(prefix="##"),
        trainer=trainers.WordPieceTrainer, 
        special_tokens={"unk": "[UNK]", "start": "[CLS]", "end": "[SEP]", 
                        "pad": "[PAD]", "mask": "[MASK]"})
    
# loads saved tokenizer
def load_tokenizer(filename):
    return Tokenizer.from_file(f"{filename}.json")

class BytePairEncodingTokenizer(BaseTokenizer):

    def __init__(self, unknown_token=None):
        super().__init__(unknown_token, model=models.BPE, normalizer=None, 
        pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False),
        decoder=decoders.ByteLevel(), trainer=trainers.BpeTrainer, 
        special_tokens={"start": "<|endoftext|>", "end": "<|endoftext|>"})

if __name__ == "__main__":
    pass


    




    
    


    

    

   


