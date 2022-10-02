import torch
import torch.nn as nn
from utils import generate_masks, generate_nopeak_pad_mask

def train(model, optimizer, dataloader, epochs=5, device=None):

    print("Training Started")
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    model.train()
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
            src_mask, tgt_mask = generate_masks(src, tgt, model.pad_idx)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            # zero the gradient
            optimizer.zero_grad()
            # get prediction and reshape outputs
            pred = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # shape: (batch_size, seq_len, vocab_size)
            pred, out = pred.view(-1, pred.size(-1)), out.contiguous().view(-1) # shape pred: (batch_size * seq_len, vocab_size) out: (batch_size * seq_len)
            # calculate loss and backpropagate
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()
            # tally loss over time
            accum_loss += loss.item()

            if (i + 1) % (m // 4) == 0:
                # diplay info every 25% of an epoch
                print(f"Epoch {epoch + 1} {(i + 1) / m * 100:.1f}% Complete | Current Loss: {accum_loss / (i + 1):.4f}")
        net_loss += accum_loss
        # display info after end of epoch
        print(f"Epoch {epoch + 1} Complete | Epoch Average Loss: {accum_loss / m:.4f}")
    net_loss /= epochs
    # display info after end of training
    print(f"Training Complete | Overall Average Loss: {net_loss:.4f}")
    return net_loss

def predict(model, src, sos_idx, eos_idx, maxlen, device=None):
    # src inshape: (batch_size, src_len,)

    # preprocess
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # encode src 
    src = torch.from_numpy(src).long().unsqueeze(0).to(device) # (1, src_len)
    src_mask = (src != model.pad_idx).unsqueeze(-2).to(device)
    # embed and positionally encoder src
    x = model.embeddings(src)
    x = model.pos_encoder(x)
    # pass through encoder layers
    e_out, attn = model.encoder(x, src_mask=src_mask)

    output = torch.tensor([[sos_idx]]).long().to(device) # generate sos

    # predict one token at a time
    while output.size(1) < maxlen:
        # decode from src and current output
        tgt_mask = generate_nopeak_pad_mask(output, model.pad_idx).to(device)
        # embed and positionally encoder output
        x = model.embeddings(output)
        x = model.pos_encoder(x)
        # pass through decoder layers
        d_out, attn1, attn2 = model.decoder(e_out, x, src_mask=src_mask, tgt_mask=tgt_mask)
        # unembed then apply softmax
        out = softmax(torch.matmul(d_out, model.wu.T))
        # get last token of highest probability
        out = torch.argmax(out, dim=-1)[:, -1].unsqueeze(0)
        # done predicting
        if out.item() == eos_idx:
            return torch.cat((output, out), dim=-1)
        # add token to current output
        output = torch.cat((output, out), dim=-1)
    
    return output.numpy()

if __name__ == "__main__":
    pass