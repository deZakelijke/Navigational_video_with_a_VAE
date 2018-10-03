import math
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class VAE(nn.Module):

    def __init__(self, latent_dims=2, image_size=(128, 128)):
        super().__init__()

        # Define layer parameters
        self.latent_dims = latent_dims
        self.chns = 3
        self.image_size = image_size
        self.filters = 32
        self.flat_dims = 512

        # Define encoding layers
        # Should I change the kernel and stride because I scaled to 128x128 instead of 64x64? Did so
        # Will revert the descision to infer the mu and sigmoid with the same conv layers.
        self.conv1_mu   = nn.Conv2d(self.chns, self.chns, (4, 4))
        self.conv2_mu   = nn.Conv2d(self.chns, self.filters, (4, 4), stride=(4, 4))
        self.conv3_mu   = nn.Conv2d(self.filters, self.filters, (6, 6), stride=(2, 2))
        self.conv4_mu   = nn.Conv2d(self.filters, self.filters, (6, 6), stride=(2, 2))
        self.fc_mu      = nn.Linear(self.flat_dims, self.latent_dims)

        self.conv1_var  = nn.Conv2d(self.chns, self.chns, (4, 4))
        self.conv2_var  = nn.Conv2d(self.chns, self.filters, (4, 4), stride=(4, 4))
        self.conv3_var  = nn.Conv2d(self.filters, self.filters, (6, 6), stride=(2, 2))
        self.conv4_var  = nn.Conv2d(self.filters, self.filters, (6, 6), stride=(2, 2))
        self.fc_logvar  = nn.Linear(self.flat_dims, self.latent_dims)

        # Define decoding layers
        self.fc_dec     = nn.Linear(self.latent_dims, self.flat_dims)
        self.deConv1    = nn.ConvTranspose2d(self.filters, self.filters, (6, 6), stride=(2, 2))
        self.deConv2    = nn.ConvTranspose2d(self.filters, self.filters, (6, 6), stride=(3, 3))
        self.deConv3    = nn.ConvTranspose2d(self.filters, self.filters, (8, 8), stride=(3, 3))
        self.deConv4    = nn.ConvTranspose2d(self.filters, self.chns, (7, 7))

        # Other network components
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout()
        self.sigmoid    = nn.Sigmoid()


    def encode(self, x):
        h1_mu  = self.relu(self.conv1_mu(x))
        h2_mu  = self.relu(self.conv2_mu(h1_mu))
        h3_mu  = self.relu(self.conv3_mu(h2_mu))
        h4_mu  = self.relu(self.conv4_mu(h3_mu))
        h5_mu  = h4_mu.view(-1, self.flat_dims)

        h1_var = self.relu(self.conv1_var(x))
        h2_var = self.relu(self.conv2_var(h1_var))
        h3_var = self.relu(self.conv3_var(h2_var))
        h4_var = self.relu(self.conv4_var(h3_var))
        h5_var = h4_var.view(-1, self.flat_dims)

        return self.sigmoid(self.fc_mu(h5_mu)), self.sigmoid(self.fc_logvar(h5_var))

    def reparametrise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h1 = self.relu(self.fc_dec(z))
        h2 = self.dropout(h1)
        h3 = h2.view(-1, 32, 4, 4)
        h4 = self.relu(self.deConv1(h3))
        h5 = self.relu(self.deConv2(h4))
        h6 = self.relu(self.deConv3(h5))
        return self.sigmoid(self.deConv4(h6))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu ,logvar


if __name__ == "__main__":
    pass
