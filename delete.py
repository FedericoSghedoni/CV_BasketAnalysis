import torch

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
print(src_data.shape, src_data)