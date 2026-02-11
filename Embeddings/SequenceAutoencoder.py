import torch
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, max_length):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(input_dim, embedding_dim),
            nn.Flatten(),
            nn.Linear(max_length * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_length * embedding_dim),
            nn.Unflatten(1, (max_length, embedding_dim)),
            nn.Linear(embedding_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded