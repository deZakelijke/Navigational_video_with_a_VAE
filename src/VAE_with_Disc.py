import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from artemis.plotting.db_plotting import dbplot

def identity(x):
    return x

class IdentityModule(nn.Module):

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PositionEstimationLayer(nn.Module):

    def __init__(self, grid_size, ranges):
        nn.Module.__init__(self)
        minrange, maxrange = ranges
        grid_data = np.array(np.meshgrid(*(np.linspace(minrange, maxrange, s) for s in grid_size))).reshape(2, -1)
        self.position_grid = torch.nn.Parameter(torch.Tensor(grid_data), requires_grad=False)  # (2, grid_size[0]*grid_size[1])
        # self.position_grid = torch.Tensor(np.array(np.meshgrid(*(np.linspace(minrange, maxrange, s) for s in grid_size))).reshape(2, -1))  # (2, grid_size[0]*grid_size[1])

    def forward(self, x):
        weights = torch.nn.functional.softmax(x, dim=1)  # (n_samples, grid_size[0]*grid_size[1])
        est_pos = (weights[:, None, :]*self.position_grid[None, :, :]).sum(dim=2)
        return est_pos


class CoordConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.coordconv = nn.Conv2d(2, out_channels, kernel_size, stride=stride, padding=padding)
        self.coords = None

    def forward(self, x):
        if self.coords is None:
            sy, sx = x.size()[-2:]
            self.coords = torch.stack(torch.meshgrid([torch.linspace(0, 1, sy), torch.linspace(0, 1, sx)]))[None].to('cuda' if torch.cuda.is_available() else 'cpu')
        return self.conv(x) + self.coordconv(self.coords)



def get_encoding_convnet(image_channels, filters, use_batchnorm, grid_size):

    flat_dim = 512 * filters//8
    grid_size_prod = grid_size[0] * grid_size[1]
    # model = nn.Sequential(
    #     nn.Conv2d(image_channels, filters, 3, stride=2, padding=1),
    #     nn.BatchNorm2d(filters) if use_batchnorm else IdentityModule(),
    #     nn.ReLU(),
    #     nn.Conv2d(filters, filters * 2, 3, stride=2, padding=1),
    #     nn.BatchNorm2d(filters * 2) if use_batchnorm else IdentityModule(),
    #     nn.ReLU(),
    #     nn.Conv2d(filters * 2, filters * 4, 3, stride=2, padding=1),
    #     nn.BatchNorm2d(filters * 4) if use_batchnorm else IdentityModule(),
    #     nn.ReLU(),
    #     Flatten(),
    #     nn.Linear(flat_dim * 4, grid_size_prod),
    #     PositionEstimationLayer(grid_size=grid_size, ranges=(-.5, .5))
    # )

    # model = resnet18(num_classes = 2)

    use_coordconv=True
    convlayer = CoordConv2d if use_coordconv else nn.Conv2d

    model = nn.Sequential(
        nn.Conv2d(image_channels, filters, kernel_size=3, padding=1),  # (64x64)
        BasicBlock(filters, filters),
        # BasicBlock(filters, filters),
        nn.Conv2d(filters, filters*4, kernel_size=3, padding=1, stride=2),  # (32x32)
        BasicBlock(filters*4, filters*4),
        # BasicBlock(filters*4, filters*4),
        nn.Conv2d(filters*4, filters*8, kernel_size=3, padding=1, stride=2),  # (16x16)
        BasicBlock(filters*8, filters*8),
        # BasicBlock(filters*8, filters*8),
        nn.Conv2d(filters*8, filters*16, kernel_size=3, padding=1, stride=2),  # (8x8)
        BasicBlock(filters*16, filters*16),
        # BasicBlock(filters*16, filters*16),
        nn.Conv2d(filters*16, filters*32, kernel_size=3, padding=1, stride=2),  # (4x4)
        Flatten(),
        nn.Linear(filters * 32 * 4 * 4, grid_size_prod),
        PositionEstimationLayer(grid_size=grid_size, ranges=(-.5, .5))

    )
    # model = ResNet(block = BasicBlock, layers = [2, 2, 2, 2])

    # ).to(device)# self.opt = torch.optim.Adam(list(self.model.parameters()), lr = learning_rate, betas = (0.5, 0.999))
    # self.opt = _get_named_opt(opt, parameters=self.model.parameters(), learning_rate=learning_rate)
    # self.position_grid = torch.Tensor(np.array(np.meshgrid(*(np.linspace(-.5, .5, s) for s in gridsize))).reshape(2, -1)).to(self.device)  # (2, grid_size[0]*grid_size[1])
    return model
    # def forward(self, x):
    #     self.model




class VAE(nn.Module):
    """ Class that combines a VAE and a GAN in one generative model

    Learns a latent epresentation for a set of images while simultaneously 
    being able to generate new sample. 
    The discriminator from the GAN model increases the quality of the results

    Args:
        latent_dims (int): number of dimensions in the latent space z
        image_size (int, int): dimensions of the image data, don't change it
    """
    def __init__(self, latent_dims=8, image_size=(64, 64), filters=32, img_channels=3, use_batchnorm = True):
        super().__init__()

        self.latent_dims = latent_dims
        self.img_chns = img_channels
        self.image_size = image_size
        self.filters = filters
        # self.flat = 512 * 4
        self.flat = 512 * filters//8
        self.intermediate_dim2 = 64 // 2 - 5
        self.intermediate_dim_disc = 32 * 60 * 60

        # Encoding layers for the mean and logvar of the latent space
        self.conv1 = nn.Conv2d(self.img_chns, self.filters, 3, stride=2, padding=1)
        self.bn_e1 = nn.BatchNorm2d(self.filters) if use_batchnorm else identity
        self.conv2 = nn.Conv2d(self.filters, self.filters * 2, 3, stride=2, padding=1)
        self.bn_e2 = nn.BatchNorm2d(self.filters * 2) if use_batchnorm else identity
        self.conv3 = nn.Conv2d(self.filters * 2, self.filters * 4, 3, stride=2, padding=1)
        self.bn_e3 = nn.BatchNorm2d(self.filters * 4) if use_batchnorm else identity
        self.fc_m  = nn.Linear(self.flat * 4, self.latent_dims)
        self.fc_s  = nn.Linear(self.flat * 4, self.latent_dims)
        self.bn_e4 = nn.BatchNorm1d(self.latent_dims) if use_batchnorm else identity


        # Decoding layers
        self.fc_d    = nn.Linear(self.latent_dims, self.flat * 4)
        self.bn_d1   = nn.BatchNorm1d(self.flat * 4) if use_batchnorm else identity
        self.deConv1 = nn.ConvTranspose2d(self.filters * 16, self.filters * 8, 3,
                                          stride=2, padding=0)
        self.bn_d2   = nn.BatchNorm2d(self.filters * 8) if use_batchnorm else identity
        self.deConv2 = nn.ConvTranspose2d(self.filters * 8, self.filters * 4, 3,
                                          stride=2, padding=1)
        self.bn_d3   = nn.BatchNorm2d(self.filters * 4) if use_batchnorm else identity
        self.deConv3 = nn.ConvTranspose2d(self.filters * 4, self.filters * 2, 3,
                                          stride=2, padding=1)
        self.bn_d4   = nn.BatchNorm2d(self.filters * 2) if use_batchnorm else identity
        self.deConv4 = nn.ConvTranspose2d(self.filters * 2, self.filters, 3,
                                          stride=2, padding=1)
        self.bn_d5   = nn.BatchNorm2d(self.filters) if use_batchnorm else identity
        self.conv_d  = nn.Conv2d(self.filters, self.img_chns, 4, 
                                 stride=1, padding=1)

        # Other network componetns
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        #print("x", x.shape)
        h1 = self.relu(self.bn_e1(self.conv1(x)))
        #print(h1.shape)
        h2 = self.relu(self.bn_e2(self.conv2(h1)))
        #print(h2.shape)
        h3 = self.relu(self.bn_e3(self.conv3(h2)))
        #print(h3.shape)
        h4 = h3.view(-1, self.flat * 4)
        #print(h4.shape)
        # mu = self.relu(self.bn_e4(self.fc_m(h4)))
        # logvar = self.relu(self.bn_e4(self.fc_s(h4)))
        # mu = self.relu(self.bn_e4(self.fc_m(h4)))
        # logvar = self.relu(self.bn_e4(self.fc_s(h4)))
        mu = self.fc_m(h4)
        logvar = self.fc_s(h4)
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
        h2 = h1.view(-1, self.flat // 4, 4, 4)
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
        mu, logvar = self.encode(x.view(-1, 3, *self.image_size))
        z_x = self.reparametrize(mu, logvar)
        recon_x = self.decode(z_x)
            
        return recon_x, mu, logvar

    # def loss_function(self, recon_x, x, mu, logvar, recon_x_disc = None, labels = None):
    #     BCE = F.binary_cross_entropy(recon_x, x, size_average = False)
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     DL  = F.binary_cross_entropy(recon_x_disc, labels)
    #     return BCE + KLD, DL
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average = False)
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





class Discriminator(nn.Module):
    """

    """
    def __init__(self, latent_dims=8, image_size=(64, 64)):
        super().__init__()

        self.filters = 64
        self.flat = 512 * 4 * 4
        self.img_chns = 3

        self.conv1 = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride = 2)
        self.conv2 = nn.Conv2d(self.filters, self.filters * 2, (2, 2), stride = 2)
        self.bn1   = nn.BatchNorm2d(self.filters * 2)
        self.conv3 = nn.Conv2d(self.filters * 2, self.filters * 4, (2, 2), stride = 2)
        self.bn2   = nn.BatchNorm2d(self.filters * 4)
        self.conv4 = nn.Conv2d(self.filters * 4, self.filters * 8, (2, 2), stride = 2)
        self.bn3   = nn.BatchNorm2d(self.filters * 8)
        self.fc    = nn.Linear(self.flat, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def discriminate(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.bn1(self.relu(self.conv2(h1)))
        h3 = self.bn2(self.relu(self.conv3(h2)))
        h4 = self.bn3(self.relu(self.conv4(h3)))
        h5 = h4.view(-1, self.flat)
        return self.sigmoid(self.fc(h5))

    def forward(self, x):
        return self.discriminate(x)

    def loss_function(self, disc_x, labels):
        BCE = F.binary_cross_entropy(disc_x, labels, size_average = False)
        return BCE



class VAEGAN:

    def __init__(self, VAE_model, GAN_model, VAE_opt, GAN_opt, latent_dims, cuda):
        self.VAE_model = VAE_model
        self.GAN_model = GAN_model
        self.VAE_opt = VAE_opt
        self.GAN_opt = GAN_opt
        self.cuda = cuda
        self.latent_dims = latent_dims

    @classmethod
    def from_init(cls, latent_dims, image_size, learning_rate, cuda):
        VAE_model = VAE(latent_dims, image_size).float()
        GAN_model = Discriminator(latent_dims, image_size).float()
        if cuda:
            VAE_model.cuda()
            GAN_model.cuda()
        VAE_opt = optim.Adam(VAE_model.parameters(), lr = learning_rate, betas = (0.5, 0.999))
        GAN_opt = optim.Adam(GAN_model.parameters(), lr = learning_rate, betas = (0.5, 0.999))
        return VAEGAN(VAE_model=VAE_model, GAN_model=GAN_model, VAE_opt=VAE_opt, GAN_opt=GAN_opt, cuda = cuda, latent_dims=latent_dims)

    def train_step(self, data):
        labels = torch.zeros(data.shape[0], 1)
        labels = Variable(labels).float()
        noise_variable = torch.zeros(data.shape[0], self.latent_dims).float().normal_()
        noise_variable = Variable(noise_variable).float()
        data = Variable(data)
        if self.cuda:
            data = data.cuda()
            labels = labels.cuda()
            noise_variable = noise_variable.cuda()

        # Optimize discriminator
        self.GAN_model.zero_grad()
        labels.fill_(1)

        predicted_real_labels  = self.GAN_model(data)
        real_GAN_loss = self.GAN_model.loss_function(predicted_real_labels, labels)
        real_GAN_loss.backward()

        gen_data = self.VAE_model.decode(noise_variable)
        labels.fill_(0)
        predicted_fake_labels = self.GAN_model(gen_data.detach())
        fake_GAN_loss = self.GAN_model.loss_function(predicted_fake_labels, labels)
        fake_GAN_loss.backward()
        self.GAN_opt.step()

        GAN_loss = real_GAN_loss.data[0] + fake_GAN_loss.data[0]

        # Optimize VAE
        self.VAE_model.zero_grad()
        recon_batch, mu, logvar = self.VAE_model(data)

        labels.fill_(1)
        predicted_gen_labels = self.GAN_model.discriminate(recon_batch)
        # rec_loss, gen_loss = self.VAE_model.loss_function(recon_batch, data, mu,
        #                                              logvar, predicted_gen_labels, labels)
        rec_loss = self.VAE_model.loss_function(recon_batch, data, mu, logvar)
        gen_loss = F.binary_cross_entropy(predicted_gen_labels, labels)
        rec_loss.backward(retain_graph = True)
        gen_loss.backward()
        self.VAE_opt.step()

        VAE_loss = rec_loss.data[0] + gen_loss.data[0]

        return VAE_loss, GAN_loss