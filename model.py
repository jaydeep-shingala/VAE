import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20, dropout=0.0):
        super().__init__()
        # encoder
        self.img_2mid = nn.Linear(input_dim, 2048)
        self.mid_3mid = nn.Linear(2048, 1024)
        self.dropout1 = nn.Dropout(dropout)
        self.img_2hid = nn.Linear(1024, h_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.mid_2img = nn.Linear(h_dim, 1024)
        self.mid3_2img = nn.Linear(1024, 2048)
        self.hid_2img = nn.Linear(2048, input_dim)
        

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.img_2mid(x))
        x = self.relu(self.mid_3mid(x))
        x = self.dropout1(x)
        h = self.relu(self.img_2hid(x))
        h = self.dropout2(h)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        h = self.dropout3(h)
        h = self.relu(self.mid_2img(h))
        h = self.relu(self.mid3_2img(h))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma*epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)