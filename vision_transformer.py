import torch
import torch.optim as optim
import numpy as np

from datasetCreator import loadDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat

# Select the videos from the dataset
dataset = loadDataset()
# print(dataset[0]['frames'][0].shape)

# Patch the images
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

# Run a quick test
sample_datapoint = torch.unsqueeze(dataset[0]['frames'][0], 0)
print("Initial shape: ", sample_datapoint.shape)
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape: ", embedding.shape)

# First implement all of the transformer building blocks.
# These blocks follows the implemantation of a classic ViT.
# Some dropouts and normalizations have been left out (evaluating the performance).

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                            num_heads=n_heads,
                                            dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    # We consider also the first
    def forward(self, x, z=torch.ones((2,2,128))):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output

# Attention(dim=128, n_heads=4, dropout=0.)(torch.ones((1, 5, 128))).shape

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# norm = PreNorm(128, Attention(dim=128, n_heads=4, dropout=0.))
# norm(torch.ones((1, 5, 128))).shape

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
# ff = FeedForward(dim=128, hidden_dim=256)
# ff(torch.ones((1, 5, 128))).shape

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# residual_att = ResidualAdd(Attention(dim=128, n_heads=4, dropout=0.))
# residual_att(torch.ones((1, 5, 128))).shape

class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                n_layers=6, out_dim=2, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                            patch_size=patch_size,
                                            emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])


model = ViT()
# To see the model
# print(model)
model(torch.ones((1, 3, 144, 144)))

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, collate_fn=lambda x: x)
test_dataloader = DataLoader(test, batch_size=32)

device = "cpu"
model = ViT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    epoch_losses = []
    model.train()
    for step, datapoint in enumerate(train_dataloader):
        # Select one video at time
        for idx, (_, _) in enumerate(datapoint):
            inputs = datapoint[idx]['frames']
            labels = torch.empty((1), dtype=int)
            labels[0] = datapoint[idx]['label']
            # Select the single frame and pass it to the model
            # We want to select the previous frame and the follower 
            # one to pass also those in the attention layer so we 
            # avoid the evaluation of the first and last frames 
            for i in range (len(inputs) - 2):
                previous = torch.unsqueeze(inputs[i].to(device), dim=0)
                input = torch.unsqueeze(inputs[i+1].to(device), dim=0)
                next = torch.unsqueeze(inputs[i+2].to(device), dim=0)
                optimizer.zero_grad()
                outputs = model(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
        epoch_losses = []
        # Something was strange when using this?
        # model.eval()
        for step, (inputs, labels) in enumerate(test_dataloader):
            for idx, (_, _) in enumerate(datapoint):
                inputs = datapoint[idx]['frames']
                labels = torch.empty((1), dtype=int)
                labels[0] = datapoint[idx]['label']
                for i in range (len(inputs) - 2):
                    previous = torch.unsqueeze(inputs[i].to(device), dim=0)
                    input = torch.unsqueeze(inputs[i+1].to(device), dim=0)
                    next = torch.unsqueeze(inputs[i+2].to(device), dim=0)
                    outputs = model(input)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
                print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)