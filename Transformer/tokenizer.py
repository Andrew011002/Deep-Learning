from transformers import AutoTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer

class WordPieceTokenizer:

    def __init__(self, vocab_size, normalizer, pre_tokenizer, decoder, special_tokens) -> None:
        self.tokenizer = self.init_tokenizer(normalizer, pre_tokenizer, decoder)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def init_tokenizer(self, normalizer, pre_tokenizer, decoder):
        tokenizer = models.WordPiece(unk_token="[UNK]")
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoder

    def train(self, corpus):
        trainer = trainers.WordPieceTrainer(self.vocab_size, special_tokens=self.special_tokens)
        Tokenizer.train_from_iterator(self.tokenizer, corpus, trainer)

    def encode(self, data):
        encoded = []

    def decode(self, data):
        decoded = []


if __name__ == "__main__":
    corpus = ["hello my name is doug", "doug is a dog who is a good boy", 
            "this is just a small example of testing the encoder", 
            "just another piece of information where we can byte-pair encode",
            "another day, another dollar you know what they say"]
    vocab_size = 100
    norms = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase(), 
                                normalizers.StripAccents(), normalizers.NFD()])
    pretoks = pre_tokenizers.Whitespace()
    deocder = decoders.WordPiece("##")
    

   


