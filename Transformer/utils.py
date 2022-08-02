import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer

class Tokenizer:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def encode(self, text, model=False):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if model:
            ids = self.tokenizer.prepare_for_model(ids)["input_ids"]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    

if __name__ == '__main__':
    tokenizer = Tokenizer()            
    sentence = "Hello, my name is Andrew and this is an example sentence. I'm not to sure how the tokenizer works so this is an attempt!"
    ids = tokenizer.encode(sentence)
    print(ids)
    original = tokenizer.decode(ids)
    print(original)

    ids = tokenizer.encode(sentence, model=True)
    print(ids)
    original = tokenizer.decode(ids)
    print(original)
