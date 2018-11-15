from __future__ import print_function

import numpy as np
import torch
import torch.utils.data

from artemis.experiments import experiment_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.image_ops import resize_image
from artemis.general.measuring_periods import measure_period
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.VAE_with_Disc import VAEGAN
from src.peters_stuff.image_crop_generator import get_image_batch_crop_generator
from src.peters_stuff.sweeps import generate_linear_sweeps


@experiment_function
def demo_train_vaegan_on_images(
        batch_size=64,
        cuda=False,
        seed=1234,
        checkpoints={0:10, 100:100, 1000: 1000},
        learning_rate=1e-3,
        latent_dims = 8,
        image_size = (64, 64),
        ):

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = VAEGAN.from_init(latent_dims=latent_dims, image_size=image_size, learning_rate=learning_rate, cuda=cuda)

    is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    cut_size = 128
    img = img[img.shape[0]//2-cut_size//2:img.shape[0]//2+cut_size//2, img.shape[1]//2-cut_size//2:img.shape[1]//2+cut_size//2]  # TODO: Revert... this is just to test on a smaller version

    dbplot(img, 'full_img')

    # for i, image_crops in enumerate(get_celeb_a_iterator(minibatch_size=batch_size, size=image_size)):
    mode = 'random'
    for i, (bboxes, image_crops) in enumerate(get_image_batch_crop_generator(img=img, crop_size=image_size, batch_size=batch_size, mode=mode, speed=10, randomness=0.1)):

        image_crops = (image_crops.astype(np.float32))/256

        var_image_crops = torch.Tensor(np.rollaxis(image_crops, 3, 1))
        if cuda:
            var_image_crops = var_image_crops.cuda()
        vae_loss, gan_loss = model.train_step(var_image_crops)

        rate = 1/measure_period('train_step')
        if do_every('5s'):
            print(f'Iter: {i}, Iter/s: {rate:.3g}, VAE-Loss: {vae_loss:.3g}, GAN-Loss: {gan_loss:.3g}')

        if is_checkpoint():
            print('Checkping')

            recons, _, _ = model.VAE_model(var_image_crops)

            z_points = torch.randn([batch_size, latent_dims]).float()
            # z_grid = torch.Tensor(np.array(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))).reshape(2, -1).T)

            n_sweep_samples = 8
            n_sweep_points = 8
            z_grid = torch.Tensor(generate_linear_sweeps(starts = np.random.randn(n_sweep_samples, latent_dims), ends=np.random.randn(n_sweep_samples, latent_dims), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, latent_dims))
            if cuda:
                z_points = z_points.cuda()
                z_grid = z_grid.cuda()

            samples = model.VAE_model.decode(z_points)
            grid_samples = model.VAE_model.decode(z_grid)

            with hold_dbplots():
                # dbplot(image_crops, 'crops')
                dbplot(np.rollaxis(recons.detach().cpu().numpy(), 1, 4), 'recons')
                dbplot(np.rollaxis(samples.detach().cpu().numpy(), 1, 4), 'samples', cornertext = f'Iter {i}')
                dbplot(np.rollaxis(grid_samples.detach().cpu().numpy().reshape((n_sweep_samples, n_sweep_points, 3)+image_size), 2, 5), 'sweeps', cornertext = f'Iter {i}')


if __name__ == '__main__':
    demo_train_vaegan_on_images(cuda=True, latent_dims=20)