import math
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, latent_dims=4, image_size=(30, 30), gamma=40):
        super().__init__()

        self.latent_dims = latent_dims
        self.image_size = image_size
        self.flat_dims = image_size[0] * image_size[1]
        self.gamma = gamma

        self.enc_1      = nn.Linear(self.flat_dims, self.flat_dims)
        self.enc_2      = nn.Linear(self.flat_dims, self.flat_dims)
        self.enc_mu     = nn.Linear(self.flat_dims, self.latent_dims)
        self.enc_logvar = nn.Linear(self.flat_dims, self.latent_dims)

        self.dec_1  = nn.Linear(self.latent_dims, self.flat_dims)
        self.dec_2  = nn.Linear(self.flat_dims, self.flat_dims)
        self.dec_3  = nn.Linear(self.flat_dims, self.flat_dims)

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def encode(self, x):
        h1 = x.view(-1, self.flat_dims)
        h2 = self.tanh(self.enc_1(h1))
        h3 = self.tanh(self.enc_2(h2))
        mu = self.enc_mu(h3)
        logvar = self.enc_logvar(h3)
        return mu, logvar

    def decode(self, z):
        h1 = self.tanh(self.dec_1(z))
        h2 = self.tanh(self.dec_2(h1))
        h3 = self.sigm(self.dec_3(h2))
        return h3.view(-1, 1, *self.image_size)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            if std.is_cuda:
                eps = torch.cuda.FloatTensor(std.data.new(std.size()).normal_())
            else:
                eps = torch.Tensor(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def permutate(self, z):
        z_new = torch.zeros(*z.shape)
        if z.is_cuda:
            z_new = z_new.cuda()
        batch_size = z.shape[0]
        for i in range(z.shape[1]):
            perm = torch.randperm(batch_size)
            z_new[:, i] = z[perm, i]
        return z_new

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        z_perm = self.permutate(z)
        return x_recon, mu, logvar, z, z_perm

    def loss(self, x, x_recon, mu, logvar, disc):
        BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
        KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        GDL = self.gamma * torch.sum(torch.log(torch.div(disc, 1 - disc)))
        return BCE - KLD - GDL


class Discriminator(nn.Module):

    def __init__(self, latent_dims=4, image_size=(30, 30)):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dims = latent_dims

        self.ln_1 = nn.Linear(self.latent_dims, self.latent_dims * 4)
        self.ln_2 = nn.Linear(self.latent_dims * 4, self.latent_dims * 4)
        self.ln_3 = nn.Linear(self.latent_dims * 4, self.latent_dims * 4)
        self.ln_4 = nn.Linear(self.latent_dims * 4, 1)

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h1 = z.view(-1, self.latent_dims)
        h2 = self.relu(self.ln_1(h1))
        h3 = self.relu(self.ln_2(h2))
        h4 = self.relu(self.ln_3(h3))
        return self.sigm(self.ln_4(h4))

    def loss(self, disc, labels):
        loss = F.binary_cross_entropy(disc, labels, reduction='sum')
        return loss
        

if __name__ == "__main__":
    pass
