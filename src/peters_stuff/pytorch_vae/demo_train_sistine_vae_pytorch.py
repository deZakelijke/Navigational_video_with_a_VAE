from __future__ import print_function

import numpy as np
import time
import torch
from torch.optim import Adam
from typing import Callable, Tuple, Iterator

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.duck import Duck
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.image_crop_generator import batch_crop, \
    iter_bbox_batches
from src.peters_stuff.pytorch_vae.convlstm import ConvLSTMPositiontoImageDecoder, ConvLSTMImageToPositionEncoder
from src.peters_stuff.pytorch_vae.pytorch_helpers import setup_cuda_if_available, get_default_device, to_default_tensor
from src.peters_stuff.pytorch_vae.pytorch_imutils import generate_random_model_path, \
    get_normed_crops_and_position_tensors, denormalize_image
from src.peters_stuff.pytorch_vae.pytorch_vaes import VAEModel
from src.peters_stuff.sample_data import SampleImages
from src.peters_stuff.sweeps import generate_linear_sweeps


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='pixel_error'), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_sistine_vae_pytorch(
        model: VAEModel,
        position_generator_constructor: Callable[[], Iterator[Tuple[int, int]]] = 'random',
        batch_size=64,
        checkpoints={0:100, 1000: 1000},
        crop_size = (64, 64),
        n_iter = None,
        save_models = False,
        ):

    img = SampleImages.sistine_512()

    cuda = setup_cuda_if_available(model)

    # optimizer = Adagrad(lr=1e-3, params = model.parameters())
    optimizer = Adam(params = model.parameters())

    dbplot(img, 'full_img')
    # model = model_constructor(batch_size, crop_size)

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)

    save_path = generate_random_model_path()

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor, n_iter=n_iter)):

        raw_image_crops, normed_image_crops, positions = get_normed_crops_and_position_tensors(img=img, bboxes=bboxes)

        signals = model.compute_all_signals(normed_image_crops)
        loss = -signals.elbo.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pixel_error = np.abs(raw_image_crops - denormalize_image(signals.x_distribution.mean)).mean()/255.

        duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=loss.item())

        if is_checkpoint():

            N_DISPLAY_IMAGES = 16

            report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, ELBO: {-loss.item():.3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)

            z_points = torch.randn(N_DISPLAY_IMAGES, model.latent_dim)
            n_sweep_samples = 8
            n_sweep_points = 8
            z_grid = to_default_tensor(generate_linear_sweeps(starts = np.random.randn(n_sweep_samples, model.latent_dim), ends=np.random.randn(n_sweep_samples, model.latent_dim), n_points=n_sweep_points).reshape(n_sweep_points*n_sweep_samples, model.latent_dim))

            samples = denormalize_image(model.decode(z_points).mean)
            grid_samples = denormalize_image(model.decode(z_grid).mean)

            with hold_dbplots():
                dbplot(raw_image_crops[:N_DISPLAY_IMAGES], 'crops')
                dbplot(denormalize_image(signals.x_distribution.mean[:N_DISPLAY_IMAGES]), 'recons')
                dbplot(samples, 'samples', cornertext = f'Iter {i}')
                dbplot(grid_samples.reshape((n_sweep_samples, n_sweep_points, *crop_size, 3)), 'sweeps', cornertext = f'Iter {i}')
                dbplot(signals.z_distribution.mean, '$\mu_z$')
                dbplot(signals.z_distribution.variance, '$\sigma^2_z$')

            save_figure_in_record()
            if save_models:
                torch.save(model, save_path)
                print(f'Model saved to {save_path}')
            yield duck


X_vae=demo_train_sistine_vae_pytorch.add_root_variant(save_models=True).add_config_variant('VAE',
    model = lambda latent_dims=20, env_hid_chans=32, dec_hid_chans=128, dec_canvas_chans=64: VAEModel(
        encoder=ConvLSTMImageToPositionEncoder(input_shape=(3, 64, 64), n_hidden_channels=env_hid_chans, n_pose_channels=latent_dims),
        decoder=ConvLSTMPositiontoImageDecoder(input_shape=(3, 64, 64), n_hidden_channels=dec_hid_chans, n_pose_channels=latent_dims, n_canvas_channels=dec_canvas_chans),
        latent_dim=latent_dims
        )
                                                                                           )


if __name__ == '__main__':

    X_vae.browse()
