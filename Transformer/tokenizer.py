
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer
from utils import create_path

def basic_tokenizer(vocab, prefix, special_tokens):
    # build tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token=special_tokens["unknown"]))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.Strip(), 
                        normalizers.NFD(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix=prefix)
    return tokenizer

class BaseTokenizer:

    def __init__(self, vocab=None, prefix=None, special_tokens=None):
        # defaults
        if prefix is None:
            prefix = "##"
        if special_tokens is None:
            special_tokens = {"start": "[CLS]", "end": "[SEP]", "unknown": "[UNK]", 
                "pad": "[PAD]", "mask": "[MASK]"}
        
        # set tokenizer & store keyword arguments
        self.tokenizer = basic_tokenizer(vocab, prefix, special_tokens)
        self.kwargs = dict(vocab=vocab, prefix=prefix, 
                    special_tokens=special_tokens)

    def train(self, corpus, size=None):
        # default
        if size is None:
            size = 30000

        # create trainer
        trainer = trainers.WordPieceTrainer(vocab_size=size, show_progress=False, 
                    special_tokens=list(self.kwargs["special_tokens"].values()), 
                    continuing_subword_prefix=self.kwargs["prefix"])

        # train & set sequence processor
        self.tokenizer.train_from_iterator(corpus, trainer)
        start, end = self.kwargs["special_tokens"]["start"], self.kwargs["special_tokens"]["end"]
        start_id, end_id = self.tokenizer.token_to_id(start), self.tokenizer.token_to_id(end)
        self.tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{start}:0 $A:0 {end}:0",
                pair=f"{start}:0 $A:0 {end}:0 $B:1 {end}:1",
                special_tokens=[(start, start_id), (end, end_id)])

    def encode(self, data, model=False, **kwargs):
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
        
    def decode(self, data, special_tokens=False, **kwargs):
        # decode the sequence(s)
        tokens = self.tokenizer.decode_batch(data, skip_special_tokens=not special_tokens)
        return tokens

    def padon(self, maxlen=None, end=True, pad_id=0):
        end = "right" if end else "left"
        pad = self.tokenizer.id_to_token(pad_id)
        self.tokenizer.enable_padding(direction=end, pad_id=self[pad],
                        pad_token=pad, length=maxlen)

    def truncon(self, maxlen, end=True):
        end = "right" if end else "left"
        self.tokenizer.enable_truncation(max_length=maxlen, direction=end)

    def inference(self):
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def vocab(self):
        return self.tokenizer.get_vocab()

    def add(self, tokens):
        self.tokenizer.add_tokens(tokens)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_special_tokens(self):
        return self.kwargs["special_tokens"]

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __getitem__(self, value):
        # try to get token or token_id of value
        if isinstance(value, int):
            return self.tokenizer.id_to_token(value)
        elif isinstance(value, str):
            vocab = self.vocab()
            return vocab.get(value, KeyError(f"{value} not in vocab"))

    def __call__(self, data, model=False, special_tokens=False, **kwargs):
        # trys to encode then decode if possible
        try: return self.encode(data, model=model)
        except: 
            try: return self.decode(data, special_tokens=special_tokens)
            except:
                raise ValueError(f"Can't interpret input {data}")

class Nerdimizer(BaseTokenizer):

    def __init__(self, vocab=None):
        super().__init__(vocab, prefix="__", special_tokens={"start": "[S]", "end": "[E]", 
                                "unknown": "[?]", "pad": "[P]", "mask": "[X]"})

class Translator:

    def __init__(self, tokenizer_encoder, tokenizer_decoder):
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder

    def encode(self, data, model=True, module=None):
        # encode based on module 
        if module == "encoder":
            return self.tokenizer_encoder.encode(data, model=model)
        if module == "decoder":
            return self.tokenizer_decoder.encode(data, model=model)
        if module is not None:
            raise ValueError(f"Unknown argument: {module}")

    def decode(self, data, special_tokens=False, module=None):
        # decode based on module
        if module == "encoder":
            return self.tokenizer_encoder.decode(data, special_tokens=special_tokens)
        if module == "decoder":
            return self.tokenizer_decoder.decode(data, special_tokens=special_tokens)
        if module is not None:
            raise ValueError(f"Unknown argument: {module}")

    def padon(self, maxlen, end=True, pad_id=0):
        self.tokenizer_encoder.padon(maxlen, end=end, pad_id=pad_id)
        self.tokenizer_decoder.padon(maxlen, end=end, pad_id=pad_id)

    def truncon(self, maxlen, end=True):
        self.tokenizer_encoder.truncon(maxlen, end=end)
        self.tokenizer_decoder.truncon(maxlen, end=end)

    def inference(self):
        self.tokenizer_encoder.inference()
        self.tokenizer_decoder.inference()

    def vocab_size(self):
        return len(self.tokenizer_encoder), len(self.tokenizer_decoder)

    def __getitem__(self, token):
        return self.tokenizer_encoder[token]

def save_tokenizer(tokenizer, path=None):
    # default
    path = "tokenizer" if path is None else path
    # create path if non-existant
    create_path(path)
    # save tokenizer to path
    tokenizer.tokenizer.save(f"{path}.json")
    print(f"Tokenizer saved")

def load_tokenizer(path=None, verbose=True):
    # default
    path = "tokenizer" if path is None else path
    # set tokenizer for base tokenizer from path
    tokenizer = BaseTokenizer()
    tokenizer.set_tokenizer(Tokenizer.from_file(f"{path}.json"))
    if verbose:
        print(f"Tokenizer loaded")
    return tokenizer

if __name__ == "__main__":
    pass


    




    


    

    

   


