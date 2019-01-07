from abc import abstractmethod
from collections.__init__ import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from src.peters_stuff.gqn_pose_predictor import convlstm_image_to_position_encoder, convlstm_position_to_image_decoder
from src.peters_stuff.tf_helpers import TFGraphClass

VAEGraph = namedtuple('VAEGraph', ['x_sample', 'z_mu', 'z_var', 'z_sample', 'x_mu', 'x_var', 'elbo', 'train_op', 'batch_size'])


class IVAEModel(object):

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    @abstractmethod
    def recon(self, x):
        raise NotImplementedError()

    @abstractmethod
    def train(self, x):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n_samples):
        raise NotImplementedError()


class TFVAEModel(TFGraphClass[VAEGraph]):
    #
    # def __init__(self, graph: VAEGraph, seed=None):
    #     self.nodes = graph
    #     tf.set_random_seed(seed)
    #     self.sess = tf.Session()
    #     self.sess.run(tf.global_variables_initializer())

    def encode(self, x):
        z_mu, z_var = self.sess.run([self.nodes.z_mu, self.nodes.z_var], feed_dict={self.nodes.x_sample: x, self.nodes.batch_size: len(x)})
        return z_mu, z_var

    def decode(self, z):
        x_mu, = self.sess.run([self.nodes.x_mu], feed_dict={self.nodes.z_sample: z, self.nodes.batch_size: len(z)})
        return x_mu

    def recon(self, x):
        x_mu, = self.sess.run([self.nodes.x_mu], feed_dict={self.nodes.x_sample: x, self.nodes.batch_size: len(x)})
        return x_mu

    def train(self, x):
        elbo, _ = self.sess.run([self.nodes.elbo, self.nodes.train_op], feed_dict={self.nodes.x_sample: x, self.nodes.batch_size: len(x)})
        return -elbo

    def sample(self, n_samples):
        x_mu, = self.sess.run([self.nodes.x_mu], feed_dict={self.nodes.z_sample: np.random.randn(n_samples, self.nodes.z_sample.shape[1]), self.nodes.batch_size: n_samples})
        return x_mu

    # def get_convlstm_constructor(self, cell_downsample=4, n_rec_maps=32, n_gen_maps=64, sequence_size=12, batch_size=64, canvas_channels=64, image_shape=(64, 64, 3), output_kernel_size=5, n_pose_channels = 2, output_type='normal', kl_scale=1.):


def get_convlstm_vae_graph(cell_downsample=4, n_rec_maps=32, n_gen_maps=64, sequence_size=12, canvas_channels=64, image_shape=(64, 64, 3), output_kernel_size=5, n_pose_channels = 2, output_type='normal', kl_scale=1.):
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Yes it's silly but we have to
    x_sample = tf.placeholder(dtype=tf.float32, shape=(None, *image_shape), name='x_sample')
    z_mu, z_var = convlstm_image_to_position_encoder(image=x_sample, batch_size=batch_size, cell_downsample=cell_downsample, n_maps=n_rec_maps, sequence_size=sequence_size, n_pose_channels=n_pose_channels)
    kl_term = 0.5 * tf.reduce_sum(z_mu**2 + z_var - 1 - tf.log(z_var), axis=1)
    z_sample = tf.random_normal(shape=(batch_size, n_pose_channels))*tf.sqrt(z_var) + z_mu
    x_params = convlstm_position_to_image_decoder(query_poses=z_sample, batch_size=batch_size, image_shape=image_shape, n_maps=n_gen_maps, canvas_channels=canvas_channels, output_kernel_size=output_kernel_size, n_pose_channels=n_pose_channels, output_type=output_type)
    if output_type=='normal':
        x_mu, x_var = x_params
        x_dist = tf.distributions.Normal(loc=x_mu, scale=tf.sqrt(x_var))
    elif output_type=='bernoulli':
        logit_mu = x_params
        x_dist = tf.distributions.Bernoulli(logits=logit_mu)
    else:
        raise Exception(output_type)
    log_likelihood = tf.reduce_sum(x_dist.log_prob(x_sample), axis=(1, 2, 3))
    elbo = tf.reduce_mean(log_likelihood - kl_scale * kl_term)
    train_op = AdamOptimizer().minimize(-elbo)
    return VAEGraph(x_sample=x_sample, z_mu=z_mu, z_var=z_var, z_sample=z_sample, x_mu = x_dist.mean(), x_var = x_dist.variance(), elbo=elbo, train_op = train_op, batch_size=batch_size)

