import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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

def train(transformer, optimizer, loss_fn, dataloader, epochs=5, device=None):

    transformer.train()

    print("Training Started")
    m = len(dataloader)
    net_loss = 0

    for epoch in range(epochs):
        
        # reset accumulative loss and display current epoch
        print(f"Epoch {epoch + 1} Started")
        accum_loss = 0

        for i, data in enumerate(dataloader):
            # get source and targets
            inputs, labels = data.to(device)
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape src: (batch_size, srclen) tgt & out: (batch_size, outlen)
            # generate the mask
            src_mask, tgt_mask = generate_mask(src, tgt, transformer.pad_idx)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # reset grad, make pred, calc loss, update params
            optimizer.zero_grad()
            pred = transformer(src, tgt, src_mask, tgt_mask, softmax=False) # shape: (batch_size, seq_len, vocab_size)
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()

            if (i + 1) % (m // 4) == 0:
                # diplay info every 1/4 of an epoch
                print(f"Epoch {epoch + 1} {np.rint((i + 1) // m * 100)}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")
        net_loss += accum_loss
        # display info after end of epoch
        print(f"Epoch {epoch + 1} Complete | Average Loss: {(i + 1) / m:.4f}")
    net_loss /= epochs
    # display info after end of training
    print(f"Training Complete | Overall Average Loss: {net_loss:.4f}")
    return net_loss / epoch
        
def create_dataloader(inputs, labels, batch_size=32, drop_last=True, shuffle=False, **dataloader_kwargs):
    # create tensors
    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
    tensorset = TensorDataset(inputs, labels)
    # create dataloader with specified args
    dataloader = DataLoader(tensorset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, **dataloader_kwargs)
    return dataloader

        
if __name__ == "__main__":
    pass
    
    


    
    
    
    





    
    
    

    







    
    



    