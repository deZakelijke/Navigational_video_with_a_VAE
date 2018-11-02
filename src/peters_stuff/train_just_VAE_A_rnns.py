from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf
import torch
import torch.utils.data
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
    iter_pos_random, batch_crop
from src.peters_stuff.sweeps import generate_linear_sweeps

VAEGraph = namedtuple('VAEGraph', ['x_sample', 'z_mu', 'z_var', 'z_sample', 'x_mu', 'x_var', 'elbo', 'train_op'])


class VAEModel(object):

    def __init__(self, graph: VAEGraph):
        self.graph = graph
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


def get_convlstm_vae_graph(cell_downsample=4, n_rec_maps=32, n_gen_maps=64, sequence_size=12, batch_size=64, canvas_channels=64, image_shape=(64, 64, 3), output_kernel_size=5, n_pose_channels = 2):
    x_sample = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_shape), name='x_sample')
    z_mu, z_var = convlstm_image_to_position_encoder(image=x_sample, cell_downsample=cell_downsample, n_maps=n_rec_maps, sequence_size=sequence_size, n_pose_channels=n_pose_channels)
    z_sample = tf.random_normal(shape=(batch_size, n_pose_channels))*z_var + z_mu
    x_mu, x_var = convlstm_position_to_image_decoder(query_poses=z_sample, image_shape=image_shape, n_maps=n_gen_maps, canvas_channels=canvas_channels, output_kernel_size=output_kernel_size, n_pose_channels=n_pose_channels)
    kl_term = -0.5 * tf.reduce_sum(1 + tf.log(z_var) - z_mu**2 - z_var, axis=1)
    log_likelihood = tf.reduce_sum(tf.distributions.Normal(loc=x_mu, scale=tf.sqrt(x_var)).log_prob(x_sample), axis=(1, 2, 3))
    elbo = tf.reduce_mean(-kl_term + log_likelihood)
    train_op = AdamOptimizer().minimize(-elbo)
    return VAEGraph(x_sample=x_sample, z_mu=z_mu, z_var=z_var, z_sample=z_sample, x_mu = x_mu, x_var = x_var, elbo=elbo, train_op = train_op )



@experiment_function
def demo_train_just_vae_on_images(
        batch_size=64,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 100:100, 1000: 1000},
        learning_rate=1e-3,
        latent_dims = 8,
        crop_size = (64, 64),
        image_cut_size = (512, 512),  # (y, x)
        n_iter = None,
        ):

    torch.manual_seed(seed)
    # if cuda:
    #     torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # model = VAETrainer(
    #     vae = VAE(latent_dims=latent_dims, image_size=image_size).to(device),
    #     learning_rate=learning_rate,
    # )
    # model = TemporallySmoothVAETrainer(
    #     vae = VAE(latent_dims=latent_dims, image_size=image_size).to(device),
    #     learning_rate=learning_rate,
    #     device=device
    # )

    model = VAEModel(
        graph = get_convlstm_vae_graph(n_pose_channels=latent_dims)
    )

    is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')



    # cut_size = 128
    # img = img[img.shape[0]//2-cut_size//2:img.shape[0]//2+cut_size//2, img.shape[1]//2-cut_size//2:img.shape[1]//2+cut_size//2]  # TODO: Revert... this is just to test on a smaller version

    img = img[img.shape[0]//2-image_cut_size[0]//2:img.shape[0]//2+image_cut_size[0]//2, img.shape[1]//2-image_cut_size[1]//2:img.shape[1]//2+image_cut_size[1]//2]  # TODO: Revert... this is just to test on a smaller version

    dbplot(img, 'full_img')

    # for i, image_crops in enumerate(get_celeb_a_iterator(minibatch_size=batch_size, size=image_size)):
    batched_bbox_generator = batchify_generator(list(
        iter_bboxes_from_positions(
            img_size=img.shape[:2],
            crop_size=crop_size,
            position_generator=iter_pos_random(n_dim=2, rng=None),
        ) for _ in range(batch_size)))

    # t_start = time.time()
    for i, bboxes in enumerate(batched_bbox_generator):
        if n_iter is not None and i>=n_iter:
            break

        image_crops = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))


        image_crops = (image_crops.astype(np.float32))/255.999-.5
        dbplot(image_crops, 'crops')

        vae_loss = model.train(image_crops)

        # var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1)).to(device)
        # if cuda:
        #     var_image_crops = var_image_crops.cuda()
        # vae_loss = model.train_step(var_image_crops)

        rate = 1/measure_period('train_step')
        if do_every('5s'):
            print(f'Iter: {i}, Iter/s: {rate:.3g}, VAE-Loss: {vae_loss:.3g}')

        if is_checkpoint():
            print('Checkping')

            # recons, _, _ = model.vae(var_image_crops)

            recons = model.recon(image_crops)




            z_points = torch.randn([batch_size, latent_dims]).float()
            # z_grid = torch.Tensor(np.array(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))).reshape(2, -1).T)

            n_sweep_samples = 8
            n_sweep_points = 8
            # z_grid = torch.Tensor(generate_linear_sweeps(starts = np.random.randn(n_sweep_samples, latent_dims), ends=np.random.randn(n_sweep_samples, latent_dims), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims))

            samples = model.decode(z_points)
            # grid_samples = model.decode(z_grid)

            with hold_dbplots():
                dbplot(image_crops, 'crops')
                dbplot(recons, 'recons')
                dbplot(samples, 'samples', cornertext = f'Iter {i}')
                # dbplot(np.rollaxis(grid_samples.detach().cpu().numpy().reshape((n_sweep_samples, n_sweep_points, 3)+image_size), 2, 5), 'sweeps', cornertext = f'Iter {i}')
                # dbplot(torch.exp(model.transition_logvar), 'transitions', plot_type='line')

if __name__ == '__main__':
    demo_train_just_vae_on_images(cuda=True, latent_dims=20)