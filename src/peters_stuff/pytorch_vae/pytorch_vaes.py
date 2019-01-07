from abc import abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Distribution, kl_divergence


class VAEModel(nn.Module):

    def __init__(self, encoder, decoder, latent_dim):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x) -> Distribution:
        return self.encoder(x)

    def decode(self, z) -> Distribution:
        return self.decoder(z)

    def prior(self) -> Distribution:
        return Normal(loc=torch.zeros(self.latent_dim), scale=1)

    def recon(self, x):
        z_distribution = self.encoder(x)
        z = z_distribution.rsample()
        distro = self.decode(z).log_prob(x)
        return distro.rsample(sample_shape = x.size())

    def elbo(self, x):
        z_distribution = self.encoder(x)
        kl_div = kl_divergence(z_distribution, self.prior())
        z = z_distribution.rsample()
        data_log_likelihood = self.decode(z).log_prob(x)
        return data_log_likelihood.flatten(1).sum(dim=1) - kl_div.sum(dim=1)

    def sample(self, n_samples):
        z = self.prior().rsample(sample_shape=(n_samples, ))
        x_dist = self.decode(z)
        return x_dist.sample()


def get_named_nonlinearity(name):
    return {'relu': nn.ReLU, 'tanh': nn.Tanh}[name]()


def make_mlp(in_size, hidden_sizes, out_size = None, nonlinearity = 'relu'):

    net = nn.Sequential()
    last_size = in_size
    for i, size in enumerate(hidden_sizes):
        net.add_module('L{}-lin'.format(i+1), nn.Linear(in_features=last_size, out_features=size))
        net.add_module('L{}-nonlin'.format(i+1),  get_named_nonlinearity(nonlinearity))
        last_size = size
    if out_size is not None:
        net.add_module('L{}-lin'.format(len(hidden_sizes)+1), nn.Linear(in_features=last_size, out_features=out_size))
    return net


class DistributionLayer(nn.Module):

    @classmethod
    def from_dense(cls, in_features, out_features):
        transform_constructor = lambda: nn.Linear(in_features, out_features)
        return cls(transform_constructor)

    @classmethod
    def from_conv(cls, in_features, out_features, kernel_size, **kwargs):
        transform_constructor = lambda: nn.Conv2d(in_features, out_features, kernel_size=kernel_size, **kwargs)
        return cls(transform_constructor)

    @staticmethod
    def get_class(name) -> 'DistributionLayer':
        return {'normal': NormalDistributionLayer, 'bernoilli': BernoulliDistributionLayer}[name]


class NormalDistributionLayer(DistributionLayer):

    def __init__(self, transform_constructor):
        super(NormalDistributionLayer, self).__init__()
        self.mean_layer = transform_constructor()
        self.logscale_layer = transform_constructor()

    def forward(self, x):
        mu = self.mean_layer(x)
        logsigma = self.logscale_layer(x)
        return Normal(loc=mu, scale=torch.exp(logsigma))


class BernoulliDistributionLayer(nn.Module):

    def __init__(self, transform_constructor):
        super(BernoulliDistributionLayer, self).__init__()
        self.logit_layer = transform_constructor()

    def forward(self, x):
        logits = self.logit_layer(x)
        return Bernoulli(logits = logits)


class NormalDistributionConvLayer(nn.Module):

    def __init__(self, transform_constructor):
        super(NormalDistributionConvLayer, self).__init__()
        self.mean_layer = transform_constructor()
        self.logscale_layer = transform_constructor()

    def forward(self, x):
        mu = self.mean_layer(x)
        logsigma = self.logscale_layer(x)
        return Normal(loc=mu, scale=torch.exp(logsigma))


class BernoulliDistributionConvLayer(nn.Module):

    def __init__(self, in_shape, out_features):
        super(BernoulliDistributionConvLayer, self).__init__()
        self.logit_layer = nn.Conv2d(in_channels=out_features)

    def forward(self, x):
        logits = self.logit_layer(x)
        return Bernoulli(logits = logits)







def make_mlp_encoder(visible_dim, hidden_sizes, latent_dim, nonlinearity ='relu'):
    net = make_mlp(in_size=visible_dim, hidden_sizes=hidden_sizes, nonlinearity=nonlinearity)
    mid_size = visible_dim if len(hidden_sizes) == 0 else hidden_sizes[-1]
    top_layer = NormalDistributionLayer(mid_size, latent_dim)
    net.add_module('z_dist', top_layer)
    return net


def make_mlp_decoder(latent_dim, hidden_sizes, visible_dim, nonlinearity ='relu', dist_type ='bernoulli'):
    net = make_mlp(in_size=latent_dim, hidden_sizes=hidden_sizes, nonlinearity=nonlinearity)
    mid_size = latent_dim if len(hidden_sizes) == 0 else hidden_sizes[-1]
    final_layer = {'normal': NormalDistributionLayer, 'bernoulli': BernoulliDistributionLayer}[dist_type](mid_size, visible_dim)
    net.add_module('output', final_layer)
    return net
