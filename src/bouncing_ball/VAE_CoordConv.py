import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

from artemis.plotting.db_plotting import dbplot


class VAE(nn.Module):
    """ Class that combines a VAE and a GAN in one generative model

    Learns a latent epresentation for a set of images while simultaneously 
    being able to generate new sample. 
    The discriminator from the GAN model increases the quality of the results

    Args:
        latent_dims (int): number of dimensions in the latent space z
        image_size (int, int): dimensions of the image data, don't change it
    """
    def __init__(self, latent_dims=8, image_size=(64, 64), filters=32, channels=3):
        super().__init__()


        # TODO make the shape of the hidden layers dependent on the image size.
        # Now its hardcoded for a 30x30 image. Uncomment old version for 64x64.

        self.latent_dims = latent_dims
        self.img_chns = channels
        self.image_size = image_size
        self.filters = filters
        # self.flat = 512 * 4
        self.flat = 512 * filters//8
        self.intermediate_dim2 = 64 // 2 - 5
        self.intermediate_dim_disc = 32 * 60 * 60

        # Encoding layers for the mean and logvar of the latent space
        self.conv1 = nn.Conv2d(self.img_chns, self.filters, 3, stride=2, padding=1)
        self.bn_e1 = nn.BatchNorm2d(self.filters)
        self.conv2 = nn.Conv2d(self.filters, self.filters * 2, 3, stride=2, padding=1)
        self.bn_e2 = nn.BatchNorm2d(self.filters * 2)
        self.conv3 = nn.Conv2d(self.filters * 2, self.filters * 4, 3, stride=2, padding=1)
        self.bn_e3 = nn.BatchNorm2d(self.filters * 4)
        self.fc_m  = nn.Linear(self.flat, self.latent_dims)
        self.fc_s  = nn.Linear(self.flat, self.latent_dims)
        self.bn_e4 = nn.BatchNorm1d(self.latent_dims)


        # Decoding layers
        self.fc_d    = nn.Linear(self.latent_dims, self.flat * 4)
        self.bn_d1   = nn.BatchNorm1d(self.flat * 4)
        #self.deConv1 = nn.ConvTranspose2d(self.filters * 16, self.filters * 8, 3,
        #                                  stride=2, padding=0)
        self.deConv1 = nn.ConvTranspose2d(self.filters * 64, self.filters * 8, 3,
                                          stride=2, padding=0)

        self.bn_d2   = nn.BatchNorm2d(self.filters * 8)
        self.deConv2 = nn.ConvTranspose2d(self.filters * 8, self.filters * 4, 3,
                                          stride=2, padding=1)
        self.bn_d3   = nn.BatchNorm2d(self.filters * 4)
        self.deConv3 = nn.ConvTranspose2d(self.filters * 4, self.filters * 2, 3,
                                          stride=2, padding=1)
        self.bn_d4   = nn.BatchNorm2d(self.filters * 2)
        self.deConv4 = nn.ConvTranspose2d(self.filters * 2, self.filters, 3,
                                          stride=2, padding=1)
        self.bn_d5   = nn.BatchNorm2d(self.filters)
        #self.conv_d  = nn.Conv2d(self.filters, self.img_chns, 4, 
        #                         stride=1, padding=1)
        self.conv_d  = nn.Conv2d(self.filters, self.img_chns, 6, 
                                 stride=1, padding=1)



        # Other network componetns
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def add_coordinate_channels(self, x):
        batch_size = x.shape[0]

        xx_ones = torch.ones([batch_size, x.shape[2]], dtype=torch.int64)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(x.shape[2]).unsqueeze(0).repeat([batch_size, 1])
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_range, xx_ones) 
        xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([batch_size, x.shape[3]], dtype=torch.int64)
        yy_ones.unsqueeze(-1)

        yy_range = torch.arange(x.shape[3]).unsqueeze(0).repeat([batch_size, 1])
        yy_range.unsqueeze(1)

        yy_channel = torch.matmul(yy_range, yy_ones) 
        yy_channel.unsqueeze(-1)

        xx_channel.float()
        yy_channel.float()
        xx_channel = xx_channel / (x.shape[2] - 1) * 2 - 1
        yy_channel = yy_channel / (x.shape[3] - 1) * 2 - 1

        ret = torch.cat([x, xx_channel, yy_channel], axis=-1)
        return ret

    def encode(self, x):
        #print("x", x.shape)
        x_with_coords = self.add_coordinate_channels(x)
        #print(f"x with coords {x_with_coords.shape})
        h1 = self.relu(self.bn_e1(self.conv1(x_with_coords)))
        #print(h1.shape)
        h2 = self.relu(self.bn_e2(self.conv2(h1)))
        #print(h2.shape)
        h3 = self.relu(self.bn_e3(self.conv3(h2)))
        #print(h3.shape)
        #h4 = h3.view(-1, self.flat * 4)
        h4 = h3.view(-1, self.flat)
        #print(h4.shape)
        mu = self.bn_e4(self.fc_m(h4))
        logvar = self.bn_e4(self.fc_s(h4))
        #print(mu.shape)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, z):
        #print("z", z.shape)

        h1 = self.relu(self.bn_d1(self.fc_d(z)))
        #print(h1.shape)
        #h2 = h1.view(-1, self.flat // 4, 4, 4)
        h2 = h1.view(-1, self.flat, 2, 2)
        #print(h2.shape)
        h3 = self.relu(self.bn_d2(self.deConv1(h2)))
        #print(h3.shape)
        h4 = self.relu(self.bn_d3(self.deConv2(h3)))
        #print(h4.shape)
        h5 = self.relu(self.bn_d4(self.deConv3(h4)))
        #print(h5.shape)
        h6 = self.relu(self.bn_d5(self.deConv4(h5)))
        #print(h6.shape)
        h7 = self.sigmoid(self.conv_d(h6))
        #print(h7.shape)
        return h7


    def forward(self, x):
        """ Feed forward function of the network

        Calculates the mean and variance of the input data. Sample z from this distribution
        and draw another random normal z. Decode both z's. Discriminate both decoded values
        and the original data x.

        Args:
            x: Input data 
        """
        mu, logvar = self.encode(x.view(-1, self.img_chns, *self.image_size))
        z_x = self.reparametrize(mu, logvar)
        recon_x = self.decode(z_x)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # DL  = F.binary_cross_entropy(recon_x_disc, labels)
        return BCE + KLD


class VAETrainer:

    def __init__(self, vae, learning_rate):
        self.vae = vae
        self.opt = optim.Adam(vae.parameters(), lr = learning_rate, betas = (0.5, 0.999))
        self.latent_dims = vae.latent_dims

    def train_step(self, data):
        # Optimize VAE
        self.vae.zero_grad()
        recon_batch, mu, logvar = self.vae(data)
        vae_loss = self.vae.loss_function(recon_batch, data, mu, logvar)
        vae_loss.backward(retain_graph = True)
        self.opt.step()
        return vae_loss



def product_of_gaussians(mu_1, var_1, mu_2, var_2):
    mu_p = (var_1**-1*mu_1 + var_2**-1*mu_2) / (var_1**-1 + var_2**-1)
    var_p = (var_1*var_2)/(var_1+var_2)
    return mu_p, var_p


class TemporallySmoothVAETrainer:

    def __init__(self, vae, learning_rate, device):
        self.vae = vae
        self.latent_dims = vae.latent_dims
        self._last_mu = None
        self._last_logvar = None
        self.transition_logvar = torch.zeros(vae.latent_dims, requires_grad=True, device=device)
        self.opt = optim.Adam(list(vae.parameters())+[self.transition_logvar], lr = learning_rate, betas = (0.5, 0.999))

    def train_step(self, data):
        # Optimize VAE
        self.vae.zero_grad()

        recon_batch, mu, logvar = self.vae(data)

        if self._last_mu is not None:
            mu, var = product_of_gaussians(
                mu_1 = self._last_mu,
                var_1 = torch.exp(self._last_logvar) + torch.exp(self.transition_logvar),
                mu_2 = mu,
                var_2 = torch.exp(logvar))


            logvar = torch.log(var)

        vae_loss = self.vae.loss_function(recon_batch, data, mu, logvar)
        vae_loss.backward(retain_graph = True)

        self._last_mu = mu.detach().requires_grad_()
        self._last_logvar = logvar.detach().requires_grad_()

        self.opt.step()
        return vae_loss

