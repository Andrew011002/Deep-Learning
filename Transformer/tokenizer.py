
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

def basic_tokenizer(vocab, prefix, special_tokens):
    # build tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token=special_tokens["unknown"]))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), 
                        normalizers.Strip()])
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
        
        # set tokenizer and store keyword arguments
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

        # train and set sequence processor
        self.tokenizer.train_from_iterator(corpus, trainer)
        start, end = self.kwargs["special_tokens"]["start"], self.kwargs["special_tokens"]["end"]
        start_id, end_id = self.tokenizer.token_to_id(start), self.tokenizer.token_to_id(end)
        self.tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{start}:0 $A:0 {end}:0",
                pair=f"{start}:0 $A:0 {end}:0 $B:1 {end}:1",
                special_tokens=[(start, start_id), (end, end_id)])

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
    def decode(self, data):
        # decode the sequence(s)
        tokens = self.tokenizer.decode_batch(data)
        return tokens

    # enables padding
    def padon(self, maxlen=None, end=True, pad_id=0):
        end = "right" if end else "left"
        pad = self.tokenizer.id_to_token(pad_id)
        self.tokenizer.enable_padding(direction=end, pad_id=self[pad],
                        pad_token=pad, length=maxlen)

    # enables truncation
    def truncon(self, maxlen, end=True):
        end = "right" if end else "left"
        self.tokenizer.enable_truncation(max_length=maxlen, direction=end)

    # disables padding and truncation for sequence inputs 
    def inference(self):
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    # returns entire vocab
    def vocab(self):
        return self.tokenizer.get_vocab()

    # adds tokens to vocabulary
    def add(self, tokens):
        self.tokenizer.add_tokens(tokens)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    # total vocabulary
    def __len__(self):
        return self.tokenizer.get_vocab_size()

    # gets id of token in vocab
    def __getitem__(self, token):
        if isinstance(token, int):
            return self.tokenizer.id_to_token(token)
        elif isinstance(token, str):
            vocab = self.vocab()
            return vocab.get(token, KeyError(f"{token} not in vocab"))

    # trys to encode then decode if possible
    def __call__(self, data, model=False):
        try: return self.encode(data, model=model)
        except: 
            try: return self.decode(data)
            except:
                raise ValueError(f"Can't interpret input {data}")

# my custom tokenizer
class Nerdimizer(BaseTokenizer):

    def __init__(self, vocab=None):
        super().__init__(vocab, prefix="__", special_tokens={"start": "[S]", "end": "[E]", 
                                "unknown": "[?]", "pad": "[P]", "mask": "[X]"})

# saves tokenizer as json
def save_tokenizer(tokenizer, path=None):
    # default
    path = "" if path is None else path

    # save tokenizer to path
    tokenizer.tokenizer.save(f"{path}.json")
    print(f"Tokenizer saved")

# loads saved tokenizer
def load_tokenizer(path=None):
    # default
    path = "" if path is None else path

    # set tokenizer for base tokenizer from path
    tokenizer = BaseTokenizer()
    tokenizer.set_tokenizer(Tokenizer.from_file(f"{path}.json"))
    print(f"Tokenizer loaded")
    return tokenizer

if __name__ == "__main__":
    pass
    


    




    


    

    

   


