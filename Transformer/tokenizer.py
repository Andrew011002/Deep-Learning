from transformers import AutoTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

class WordPieceTokenizer:

    def __init__(self, vocab_size, normalizer, pre_tokenizer, prefix, special_tokens) -> None:
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.special_tokens = special_tokens
        self.tokenizer = self.init_tokenizer(normalizer, pre_tokenizer, prefix)

    def init_tokenizer(self, normalizer, pre_tokenizer, prefix):
        # build basic structure of WPT
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.special_tokens["unknown"]))
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoders.WordPiece(prefix)
        return tokenizer

    def init_post_processor(self):
        # get sos/eos tokens and ids
        sos, eos = self.special_tokens["sos"], self.special_tokens["eos"]
        ids = self.get_vocab() 
        # structure how tokens are process for single sentence and sentence pair
        post_processor = processors.TemplateProcessing(single=f"{sos} $A:0 {eos}:0",
                                                        pair=f"{sos} $A:0 {eos}:0 $B:1 {eos}:1", 
                                                        special_tokens=[(sos, ids[sos]), (eos, ids[eos])])
        self.tokenizer.post_processor = post_processor

    def train(self, corpus):
        # train tokenizer to desired vocab size w/ special tokens and desired prefix
        trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=list(self.special_tokens.values()), 
                                            continuing_subword_prefix=self.prefix)
        Tokenizer.train_from_iterator(self.tokenizer, corpus, trainer)
        # init the post processor for formatting structure of encoded sequences
        self.init_post_processor()

    def encode(self, data, tokens_only=False):
        encoded = []
        # encode all sequences to map of metadata
        for sequence in data:
            # encode sequence pair
            if isinstance(sequence, tuple):
                encoding = self.tokenizer.encode(*sequence)
            # encode single sentence
            else:
                encoding = self.tokenizer.encode(sequence)
            # only add tokens (ignore other metadata)
            if tokens_only:
                encoded.append(encoding.tokens)
            # keep all metadata
            else:
                encoded.append(encoding)
        return encoded 

    def decode(self, data):
        decoded = []
        # decode all encoded token ids
        for encoding in data:
            sequence = self.tokenizer.decode(encoding.ids)
            decoded.append(sequence)
        return sequence

    def get_vocab(self):
        return self.tokenizer.get_vocab()


if __name__ == "__main__":
    corpus = ["hello my name is doug", "doug is a dog who is a good boy", 
            "this is just a small example of testing the encoder", 
            "just another piece of information where we can byte-pair encode",
        "another day, another dollar you know what they say"]
    vocab_size = 100
    norms = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase(), 
                                normalizers.StripAccents(), normalizers.NFD()])
    pretoks = pre_tokenizers.Whitespace()
    special_tokens = dict(unknown="[UNK]", sos="[SOS]", eos="[EOS]", 
                        pad="[PAD]", mask="[MASK]")
    tokenizer = WordPieceTokenizer(vocab_size, norms, pretoks, "##", special_tokens=special_tokens)
    tokenizer.train(corpus)

    

   


