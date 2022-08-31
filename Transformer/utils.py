import torch
import torch.nn as nn
import numpy as np
from sublayers import ScaledDotProductAttention
from embedding import Embeddings

def generate_mask(inputs, outputs, pad_idx):
    # inshape inputs: (batch_size, inputs_len) outputs: (batch_size, outputs_len) pad_idx: (,)
    tgt_len = outputs.size(1)

    # create padded mask for src & tgt 
    src_mask = (inputs != pad_idx).unsqueeze(-2)
    tgt_mask = (outputs != pad_idx).unsqueeze(-2)

    # create subsequent mask for tgt (no peak) shape tgt_nopeak_mask: (1, seq_len - 1, seq_len - 1)
    tgt_nopeak_mask = np.triu(np.ones((1, tgt_len, tgt_len)), k=1).astype(int)
    tgt_nopeak_mask = torch.autograd.Variable(torch.from_numpy(tgt_nopeak_mask) == 0)

    # combine tgt_pad_mask & tgt_nopeak_mask to hide pad and prevent subsequent attention
    tgt_mask = tgt_mask & tgt_nopeak_mask
    return src_mask, tgt_mask

    


if __name__ == "__main__":
    batch_size = 3
    vocab_size = 1000
    max_len = 5
    pad_idx = 0
    dm = 512

    embeddings = Embeddings(vocab_size, dm, pad_idx)
    scaled_dot_prod = ScaledDotProductAttention(dm)

    inputs = torch.randint(0, vocab_size, (batch_size, max_len))
    outputs = inputs[:, :-1]

    src = embeddings(inputs)
    tgt = embeddings(outputs)
    src_mask, tgt_mask = generate_mask(inputs, outputs, pad_idx)
    print(inputs)
    context_src, attn_src = scaled_dot_prod(src, src, src, src_mask)
    print(outputs)
    context_tgt, attn_tgt = scaled_dot_prod(tgt, tgt, tgt, tgt_mask)
    
    

    







    
    



    