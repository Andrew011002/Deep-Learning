import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# data configurations
train_size = 100000
test_size = 1000
vocab_size_english = 32000
vocab_size_german = 32000
maxlen = 10 # TODO
batch_size = 128

# model configurations
dm = 512
nhead = 8
layers = 6
dff = 2048
bias = False
dropout = 0.1
eps = 1e-5

# optimizer configurations
adam_eps = 10e-9
lr = 1e-5
betas = (0.9, 0.98)

# scheduler
factor = 0.9
patience = 10

# decoder search configurations
beam_width = 3
max_breadth = 100
search_mode = "best"
alpha = 0.6

# training & metric configurations
sample_size = 10
goal_bleu = 23
ngrams = 4
save_every = 5
overwrite = False
warmups = 100
epochs = 1000
verbose = True

