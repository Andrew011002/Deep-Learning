import torch
import torch.nn as nn
from torch import LongTensor
from utils import generate_mask

def train(model, optimizer, dataloader, epochs=5, device=None):

    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    

    print("Training Started")
    m = len(dataloader)
    net_loss = 0

    for epoch in range(epochs):
        
        # reset accumulative loss and display current epoch
        print(f"Epoch {epoch + 1} Started")
        accum_loss = 0

        for i, data in enumerate(dataloader):
            # get source and targets
            inputs, labels = data
            src, tgt, out = inputs, labels[:, :-1], labels[:, 1:] # shape src: (batch_size, srclen) tgt & out: (batch_size, outlen)
            src, tgt, out = src.long().to(device), tgt.long().to(device), out.long().to(device)
            # generate the mask
            src_mask, tgt_mask = generate_mask(src, tgt, model.pad_idx)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # zero grad, make pred, calc loss, update params
            optimizer.zero_grad()
            pred = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # shape: (batch_size, seq_len, vocab_size)
            pred, out = pred.view(-1, pred.size(-1)), out.contiguous().view(-1) # shape pred: (batch_size * seq_len, vocab_size) out: (batch_size * seq_len)
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()

            if (i + 1) % (m // 4) == 0:
                # diplay info every 1/4 of an epoch
                print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")
        net_loss += accum_loss
        # display info after end of epoch
        print(f"Epoch {epoch + 1} Complete | Epoch Average Loss: {accum_loss / m:.4f}")
    net_loss /= epochs
    # display info after end of training
    print(f"Training Complete | Overall Average Loss: {net_loss:.4f}")
    return net_loss


def predict(model, sos_idx, eos_idx, source, prompt=False):


    # encode sequence

    # give decoder sos

    # pred from encoded sequence and sos

    # apply softmax to last dim

    # add pred to sos 

    # rerun until it's maxlen or eos token




    if __name__ == "__main__":
        pass