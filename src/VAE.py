import math
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms


class VAE(nn.Module):

    def __init__(self, latent_dims=2):
        super().__init__()

        # Define layer parameters
        self.latent_dims = latent_dims
        self.chns = 3
        self.image_size = (128, 128)
        self.filers = 32
        self.flat_dims = 10

        # Define encoding layers
        # Should I change the kernel and stride because I scaled to 128x128 instead of 64x64?
        self.conv1      = nn.Conv2d(self.img_chns, self.img_chns, (2, 2))
        self.conv2      = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride=(2, 2))
        self.conv3      = nn.Conv2d(self.filters, self.filters, (3, 3), stride=(1, 1))
        self.conv4      = nn.Conv2d(self.filters, self.filters, (3, 3), stride=(1, 1))
        self.fc_mu      = nn.Linear(self.flat_dims, self.latents_dims)
        self.fc_logvar  = nn.Linear(self.flat_dims, self.latents_dims)

        # Define decoding layers
        self.fc_dec     = nn.Linear(self.latent_dims, self.flat_dims)
        self.deConv1    = nn.ConvTranspose2d(self.filters, self.filters, (3, 3), stride=(1, 1))
        self.deConv2    = nn.ConvTranspose2d(self.filters, self.filters, (3, 3), stride=(1, 1))
        self.deConv3    = nn.ConvTranspose2d(self.filters, self.filters, (3, 3), stride=(2, 2))
        self.deConv4    = nn.ConvTranspose2d(self.filters, self.chns, 2)

        # Other network components
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout()
        self.sigmoid    = nn.Sigmoid()


    def encode(self, x):
        h1  = self.relu(self.conv1(x))
        h2  = self.relu(self.conv2(h1))
        h3  = self.relu(self.conv3(h2))
        h4  = self.relu(self.conv4(h3))
        print(h4.shape)
        h5  = h4.view(-1, self.flat_dims)
        return self.sigmoid(self.fc_mu(h5)), self.sigmoid(self.fc_logvar(h5))

    def reparametrise(self, mu, logvar):
        return mu

    def decode(self, z):
        pass

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu ,logvar


if __name__ == "__main__":
    pass
