from transformers import AutoTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

class WordPieceTokenizer:

    def __init__(self, vocab_size, normalizer, pre_tokenizer, prefix, special_tokens, maxlen=None) -> None:
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.special_tokens = special_tokens
        self.tokenizer = self.init_tokenizer(normalizer, pre_tokenizer, prefix)
        self.maxlen = maxlen

    def init_tokenizer(self, normalizer, pre_tokenizer, prefix):
        # build basic structure of WPT
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.special_tokens["unk"]))
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoders.WordPiece(prefix)
        return tokenizer

    def init_post_processor(self):
        # get cls/sep tokens and ids
        cls, sep = self.special_tokens["cls"], self.special_tokens["sep"]
        ids = self.vocab() 
        # structure how tokens are process for single sentence and sentence pair
        post_processor = processors.TemplateProcessing(single=f"{cls} $A:0 {sep}:0",
                                                        pair=f"{cls} $A:0 {sep}:0 $B:1 {sep}:1", 
                                                        special_tokens=[(cls, ids[cls]), (sep, ids[sep])])
        self.tokenizer.post_processor = post_processor
        
        # enable padding and truncation for encodings
        if self.maxlen is not None:
            pad_id = self.vocab().get(self.special_tokens["pad"])
            self.tokenizer.enable_padding(pad_id=pad_id, length=self.maxlen)
            self.tokenizer.enable_truncation(max_length=self.maxlen)

    def train(self, corpus):
        # train tokenizer to desired vocab size w/ special tokens and desired prefix
        trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=list(self.special_tokens.values()), 
                                            continuing_subword_prefix=self.prefix)
        Tokenizer.train_from_iterator(self.tokenizer, corpus, trainer)
        # init the post processor for formatting structure of encoded sequences
        self.init_post_processor()

    def inference(self):
        # disable padding and truncation
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def encode(self, data, model=False):
        encoded = []
        # encode all sequences to map of metadata
        for sequence in data:
            # encode sequence pair
            if isinstance(sequence, tuple):
                encoding = self.tokenizer.encode(*sequence)
            # encode single sentence
            else:
                encoding = self.tokenizer.encode(sequence)
            # add ids for model training
            if model:
                encoded.append(encoding.ids)
            # not for model add tokens
            else:
                encoded.append(encoding.tokens)
        return encoded 

    def decode(self, data):
        decoded = []
        # decode all encoded token ids
        for encoding in data:
            sequence = self.tokenizer.decode(encoding)
            decoded.append(sequence)
        return sequence

    def vocab(self):
        return self.tokenizer.get_vocab()

    def save(self, filename):
        self.tokenizer.save(f"{filename}.json")

def load_tokenizer(filename):
    return Tokenizer.from_file(f"{filename}.json")


if __name__ == "__main__":
    corpus = ["hello my name is doug", "doug is a dog who is a good boy", 
            "this is just a small example of testing the encoder", 
            "just another piece of information where we can byte-pair encode",
        "another day, another dollar you know what they say"]
    vocab_size = 100
    norms = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase(), 
                                normalizers.StripAccents(), normalizers.NFD()])
    pretoks = pre_tokenizers.Whitespace()
    special_tokens = dict(unk="[UNK]", cls="[CLS]", sep="[SEP]", 
                        pad="[PAD]", mask="[MASK]")
    tokenizer = WordPieceTokenizer(vocab_size, norms, pretoks, "##", special_tokens=special_tokens, maxlen=100)

    
    


    

    

   


