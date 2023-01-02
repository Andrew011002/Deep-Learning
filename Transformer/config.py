import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# data configurations
train_size = 100000
test_size = 1000
vocab_size_english = 25000
vocab_size_german = 25000
maxlen = 256
batch_size = 128

# model configurations
dm = 512
nhead = 8
layers = 6
dff = 2048
bias = False
dropout = 0.1
eps = 1e-5
adam_eps = 10e-9
lr = 1e-5
betas = (0.9, 0.98)
factor = 0.9
patience = 10

# training & metric configurations
ngrams = 4
save_every = 5
warmups = 100
epochs = 1000
verbose = True

