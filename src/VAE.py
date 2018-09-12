import math
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms


class VAE(nn.Module):

    def __init__(self, latent_dims=2):
        super().__init__()

        # Define layers
        self.latent_dims = latent_dims
        self.chns = 3
        self.image_size = (64, 64)



    def encode(self, x):
        pass

    def reparametrise(self, mu, logvar):
        pass

    def decode(self, z):
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    pass
