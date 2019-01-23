import math
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, latent_dims=2, image_size=(30, 30), supervision=1.0, desired_dims=2):
        super().__init__()

        self.latent_dims = latent_dims
        self.image_size = image_size
        self.intermediate_dims = 160
        self.flat_dims = 30 * 30
        self.supervision = supervision
        self.desired_dims = desired_dims

        self.enc1    = nn.Linear(self.flat_dims, self.intermediate_dims)
        self.enc2    = nn.Linear(self.intermediate_dims, self.intermediate_dims)
        self.enc_mu  = nn.Linear(self.intermediate_dims, self.latent_dims)
        self.enc_std = nn.Linear(self.intermediate_dims, self.latent_dims)

        self.dec1    = nn.Linear(self.latent_dims, self.intermediate_dims)
        self.dec2    = nn.Linear(self.intermediate_dims, self.intermediate_dims)
        self.dec_mu  = nn.Linear(self.intermediate_dims, self.flat_dims)
        self.dec_std = nn.Linear(self.intermediate_dims, self.flat_dims)

        self.activation = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def encode(self, x):
        reshape = x.view(-1, 900)
        h1 = self.activation(self.enc1(reshape))
        h2 = self.activation(self.enc2(h1))
        h3 = self.enc_mu(h2)
        h4 = self.enc_std(h2)
        return h3, h4

    def decode(self, z):
        h5 = self.activation(self.dec1(z))
        h6 = self.activation(self.dec2(h5))
        h7 = self.sigm(self.dec_mu(h6))
        #h8 = self.sigm(self.dec_std(h6))
        return h7

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        mu2 = self.decode(z)
        #x_hat = (self.reparametrise(mu2, logvar2)).view(-1, 1, 30, 30)
        x_hat = mu2.view(-1, 1, 30, 30)
        return x_hat, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def loss_function_with_SSV(self, recon_x, x, mu, logvar, position):
        """ Calculates three loss values and returns their sum
            
        Calculates the Binary Cross entropy between the ground truth and 
        the generated samples. 
        Calulates the KL-divergence of the latent
        representation. 
        Ca
        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = (1 - self.supervision) * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        PDL = self.supervision * (F.mse_loss(mu[:, :self.desired_dims], position, reduction="sum")
            -0.5 * torch.sum(1 + logvar[:, self.desired_dims:] - mu[:, self.desired_dims:].pow(2) 
            - logvar[:, self.desired_dims:].exp()))
        return BCE + KLD + PDL
          

if __name__ == "__main__":
    pass
