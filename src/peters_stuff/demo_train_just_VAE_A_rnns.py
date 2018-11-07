from __future__ import print_function

from abc import abstractmethod
from collections import namedtuple

import numpy as np
import tensorflow as tf
import torch
import torch.utils.data
from artemis.experiments.decorators import ExperimentFunction

from artemis.experiments.experiment_record import save_figure_in_record

from artemis.plotting.saving_plots import save_figure

from artemis.general.should_be_builtins import bad_value

from artemis.general.duck import Duck
from tensorflow.python.training.adam import AdamOptimizer

from artemis.experiments import experiment_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image
from artemis.general.measuring_periods import measure_period
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.gqn_pose_predictor import convlstm_image_to_position_encoder, convlstm_position_to_image_decoder
from src.peters_stuff.image_crop_generator import iter_bboxes_from_positions, \
    iter_pos_random, batch_crop, iter_bbox_batches
from src.peters_stuff.sweeps import generate_linear_sweeps

VAEGraph = namedtuple('VAEGraph', ['x_sample', 'z_mu', 'z_var', 'z_sample', 'x_mu', 'x_var', 'elbo', 'train_op'])


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


class TFVAEModel(IVAEModel):

    def __init__(self, graph: VAEGraph, seed=None):
        self.graph = graph
        tf.set_random_seed(seed)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def encode(self, x):
        z_mu, z_var = self.sess.run([self.graph.z_mu, self.graph.z_var], feed_dict={self.graph.x_sample: x})
        return z_mu, z_var

    def decode(self, z):
        x_mu, = self.sess.run([self.graph.x_mu], feed_dict={self.graph.z_sample: z})
        return x_mu

    def recon(self, x):
        x_mu, = self.sess.run([self.graph.x_mu], feed_dict={self.graph.x_sample: x})
        return x_mu

    def train(self, x):
        elbo, _ = self.sess.run([self.graph.elbo, self.graph.train_op], feed_dict={self.graph.x_sample: x})
        return -elbo

    def sample(self, n_samples):
        x_mu, = self.sess.run([self.graph.x_mu], feed_dict={self.graph.z_sample: np.random.randn(n_samples, self.graph.z_sample.shape[1])})
        return x_mu


def get_convlstm_vae_graph(cell_downsample=4, n_rec_maps=32, n_gen_maps=64, sequence_size=12, batch_size=64, canvas_channels=64, image_shape=(64, 64, 3), output_kernel_size=5, n_pose_channels = 2, output_type='normal', kl_scale=1.):
    x_sample = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_shape), name='x_sample')
    z_mu, z_var = convlstm_image_to_position_encoder(image=x_sample, cell_downsample=cell_downsample, n_maps=n_rec_maps, sequence_size=sequence_size, n_pose_channels=n_pose_channels)
    kl_term = 0.5 * tf.reduce_sum(z_mu**2 + z_var - 1 - tf.log(z_var), axis=1)
    z_sample = tf.random_normal(shape=(batch_size, n_pose_channels))*tf.sqrt(z_var) + z_mu
    x_params = convlstm_position_to_image_decoder(query_poses=z_sample, image_shape=image_shape, n_maps=n_gen_maps, canvas_channels=canvas_channels, output_kernel_size=output_kernel_size, n_pose_channels=n_pose_channels, output_type=output_type)
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
    return VAEGraph(x_sample=x_sample, z_mu=z_mu, z_var=z_var, z_sample=z_sample, x_mu = x_dist.mean(), x_var = x_dist.variance(), elbo=elbo, train_op = train_op )



@ExperimentFunction(one_liner_function=lambda result: f'Loss: {result[-len(result)//10:, "loss"].to_array().mean():.3g}')
def demo_train_just_vae_on_images(
        batch_size=64,
        checkpoints={0:10, 100:100, 1000: 400},
        latent_dims = 2,
        kl_scale = 1.,
        crop_size = (64, 64),
        image_cut_size = (512, 512),  # (y, x)
        # image_cut_size = (128, 128),  # (y, x)
        n_iter = None,
        output_type = 'bernoulli',
        seed=1234,
        data_seed = 1235

        ):

    model = TFVAEModel(
        graph = get_convlstm_vae_graph(n_pose_channels=latent_dims, output_type = output_type, kl_scale=kl_scale),
        seed=seed
    )
    data_rng = np.random.RandomState(data_seed)

    is_checkpoint = Checkpoints(checkpoints)
    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    img = img[img.shape[0]//2-image_cut_size[0]//2:img.shape[0]//2+image_cut_size[0]//2, img.shape[1]//2-image_cut_size[1]//2:img.shape[1]//2+image_cut_size[1]//2]  # TODO: Revert... this is just to test on a smaller version

    dbplot(img, 'full_img')

    data = Duck()

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor='random', rng=data_rng)):

        if n_iter is not None and i>=n_iter:
            break
        image_crops = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))
        image_crops = \
            (image_crops.astype(np.float32))/255.999-.5 if output_type=='normal' else \
            (image_crops.astype(np.float32))/255.999 if output_type=='bernoulli' else \
            bad_value(output_type)

        vae_loss = model.train(image_crops)
        data[next, :] = dict(loss = vae_loss)

        rate = 1/measure_period('train_step')
        if do_every('5s'):
            print(f'Iter: {i}, Iter/s: {rate:.3g}, VAE-Loss: {vae_loss:.3g}')

        if is_checkpoint():
            print('Checkping')

            recons = model.recon(image_crops)
            z_points = np.random.randn(batch_size, latent_dims)
            z_enc_mu, z_enc_var = model.encode(image_crops)
            n_sweep_samples = 8
            n_sweep_points = 8
            z_grid = generate_linear_sweeps(starts = np.random.randn(n_sweep_samples, latent_dims), ends=np.random.randn(n_sweep_samples, latent_dims), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims)

            samples = model.decode(z_points)
            grid_samples = model.decode(z_grid)

            with hold_dbplots():
                dbplot(image_crops, 'crops')
                dbplot(recons, 'recons')
                dbplot(samples, 'samples', cornertext = f'Iter {i}')
                dbplot(grid_samples.reshape((n_sweep_samples, n_sweep_points, *crop_size, 3)), 'sweeps', cornertext = f'Iter {i}')
                dbplot(data.to_array(), 'loss', plot_type='line')
                dbplot(z_enc_mu, '$\mu_z$')
                dbplot(z_enc_var, '$\sigma^2_z$')
            save_figure_in_record()

            yield data


X_trial = demo_train_just_vae_on_images.add_root_variant(n_iter=10000)

X_trial.add_variant(output_type='normal', latent_dims=20, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=20, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=3, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=2, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=20, kl_scale=10.)

if __name__ == '__main__':
    # demo_train_just_vae_on_images.get_variant('search').run()
    demo_train_just_vae_on_images.browse()
