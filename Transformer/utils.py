import torch
import torch.nn as nn
from sublayers import MultiHeadAttention
from embedding import Embeddings

def generate_mask(inputs, outputs, pad_idx):
    # inshape inputs: (batch_size, inputs_len) outputs: (batch_size, outputs_len) pad_idx: (,)
    tgt_len = outputs.size(1)

    # create padded mask for src & tgt 
    src_mask = (inputs != pad_idx).unsqueeze(-2)
    tgt_mask = (outputs != pad_idx).unsqueeze(-2)

    # create subsequent mask for tgt (no peak) shape tgt_nopeak_mask: (1, tgt_len, tgt_len)
    tgt_nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len)) == 1)
    tgt_nopeak_mask = tgt_nopeak_mask.transpose(1, 2)
    
    # combine tgt_pad_mask & tgt_nopeak_mask to hide pad and prevent subsequent attention
    tgt_mask = tgt_mask & tgt_nopeak_mask
    # shape src_mask: (batch_size, 1, seq_len) tgt_mask: (batch_size, tgt_len, tgt_len)
    return src_mask, tgt_mask

def train():
    pass

if __name__ == "__main__":
    batch_size = 2
    vocab_size = 10
    max_len = 5
    pad_idx = 0
    dm = 512

    inputs = torch.randint(0, vocab_size, (batch_size, max_len))
    outputs = inputs[:, :-1]
    src_mask, tgt_mask = generate_mask(inputs, outputs, pad_idx)

    embeddings = Embeddings(vocab_size, dm, pad_idx)
    src = embeddings(inputs)
    tgt = embeddings(outputs)
    multihead = MultiHeadAttention(512, 8)
    context, attn = multihead(src, src, src, src_mask)
    context, attn = multihead(tgt, tgt, tgt, tgt_mask)



    
    
    

    







    
    



    