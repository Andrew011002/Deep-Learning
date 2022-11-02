from transformers import AutoTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers,\
processors, trainers, Tokenizer


class WordPieceTokenizer:

    def __init__(self, normalizer, pre_tokenizer, model, processor) -> None:
        pass


if __name__ == "__main__":
    corpus = ["hello my name is doug", "doug is a dog who is a good boy", 
            "this is just a small example of testing the encoder", 
            "just another piece of information where we can byte-pair encode",
            "another day, another dollar you know what they say"]

   


