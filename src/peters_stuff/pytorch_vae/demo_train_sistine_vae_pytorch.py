from __future__ import print_function

import numpy as np
import time
import torch
from torch.optim import Adam
from typing import Callable, Tuple, Iterator, Union

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.duck import Duck
from artemis.general.should_be_builtins import bad_value, switch
from artemis.ml.parameter_schedule import ParameterSchedule
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from src.peters_stuff.image_crop_generator import batch_crop, \
    iter_bbox_batches
from src.peters_stuff.pytorch_vae.convlstm import ConvLSTMPositiontoImageDecoder, ConvLSTMImageToPositionEncoder
from src.peters_stuff.pytorch_vae.pytorch_helpers import setup_cuda_if_available, get_default_device, to_default_tensor
from src.peters_stuff.pytorch_vae.pytorch_imutils import generate_random_model_path, \
    get_normed_crops_and_position_tensors, denormalize_image
from src.peters_stuff.pytorch_vae.pytorch_vaes import VAEModel, get_named_optimizer, DistributionLayer, \
    get_supervised_vae_loss
from src.peters_stuff.sample_data import SampleImages
from src.peters_stuff.sweeps import generate_linear_sweeps


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield=['pixel_error', 'elbo', 'kl']), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_sistine_vae_pytorch(
        model: VAEModel,
        position_generator_constructor: Union[str, Callable[[], Iterator[Tuple[int, int]]]] = 'normal',
        batch_size=64,
        checkpoints={0:100, 1000: 1000},
        crop_size = (64, 64),
        # n_iter = None,
        n_iter = 30000,  # 10000 iterations is about an hour.
        save_models = False,
        optimizer = ('adam', {}),
        supervision_schedule = 0,
        z_sample_schedule = 'natural',
        zero_irrelevant_latents_schedule = False,  #
        normscale = 0.25,  # How concentrated the samples are in the middle of the image (smaller -> more concentration)
        ):

    img = SampleImages.sistine_512()

    cuda = setup_cuda_if_available(model)

    # assert zero_init_irrelevant_latents in (True, False, 'always')
    # if zero_init_irrelevant_latents in (True, 'always'):
    #     next(model.decoder.parameters()).data[:, 2:, :, :] = 0


    optimizer = get_named_optimizer(name = optimizer[0], params=model.parameters(), args=optimizer[1])

    dbplot(img, 'full_img')
    # model = model_constructor(batch_size, crop_size)

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)

    save_path = generate_random_model_path()

    if supervision_schedule is not None:
        supervision_schedule = ParameterSchedule(supervision_schedule)

    z_sample_schedule = ParameterSchedule(z_sample_schedule)
    zero_irrelevant_latents_schedule = ParameterSchedule(zero_irrelevant_latents_schedule)

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor, n_iter=n_iter, normscale=normscale)):

        raw_image_crops, normed_image_crops, positions = get_normed_crops_and_position_tensors(img=img, bboxes=bboxes, scale=1./normscale)

        with switch(z_sample_schedule(i)) as case:
            z_samples = \
                None if case('natural') else \
                'no_reparametrization' if case('no_reparametrization') else \
                torch.cat([positions, torch.zeros((batch_size, model.latent_dim-2))], dim=1) if case('target', 'target_kl') else \
                bad_value(case.value)

            signals = model.compute_all_signals(normed_image_crops, z_samples = z_samples)

            supervision_factor = supervision_schedule(i) if supervision_schedule is not None else None
            if supervision_factor:
                loss = get_supervised_vae_loss(signals=signals, target=positions, supervision_factor=supervision_factor, kl_on_targets=case('target_kl')).mean()
            else:
                loss = -signals.elbo.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if zero_irrelevant_latents_schedule(i):
            next(model.decoder.parameters()).data[:, 2:, :, :] = 0

        # pixel_error = np.abs(raw_image_crops - denormalize_image(signals.x_distribution.mean)).mean()/255.

        duck[next, :] = dict(
            iter=i,
            pixel_error=np.abs(raw_image_crops - denormalize_image(signals.x_distribution.mean)).mean()/255.,
            kl = signals.kl_div.mean(),
            data_logp = signals.data_log_likelihood.mean(),
            elapsed=time.time()-t_start,
            training_loss=loss.item(),
            elbo=signals.elbo.mean().item(),
            supervision_factor=supervision_factor
            )

        if is_checkpoint():

            N_DISPLAY_IMAGES = 16

            lastduck = duck[-1]
            report = f"Iter: {i}, Pixel Error: {lastduck['pixel_error']:3g}, ELBO: {lastduck['elbo']:.3g}, KL: {lastduck['kl']:.3g}, Data LogP: {lastduck['data_logp']:.3g}, Training Loss: {lastduck['training_loss']:.3g}, Mean Rate: {i/lastduck['elapsed']:.3g}iter/s, Supervision Factor: {supervision_factor:.3g}"
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
                dbplot((positions[:, 0].detach().cpu().numpy(), positions[:, 1].detach().cpu().numpy()), 'positions', plot_type=DBPlotTypes.SCATTER)
                dbplot((signals.z_distribution.mean[:, 0].detach().cpu().numpy(), signals.z_distribution.mean[:, 1].detach().cpu().numpy()), 'predicted_positions', plot_type=DBPlotTypes.SCATTER)

            save_figure_in_record()
            if save_models:
                torch.save(model, save_path)
                print(f'Model saved to {save_path}')
            yield duck


X_vae=demo_train_sistine_vae_pytorch.add_root_variant(save_models=True).add_config_variant('VAE',
    model = lambda latent_dims=20, env_hid_chans=32, dec_hid_chans=128, dec_canvas_chans=64: VAEModel(
        encoder=ConvLSTMImageToPositionEncoder(input_shape=(3, 64, 64), n_hidden_channels=env_hid_chans, n_pose_channels=latent_dims),
        decoder=ConvLSTMPositiontoImageDecoder(input_shape=(3, 64, 64), n_hidden_channels=dec_hid_chans, n_pose_channels=latent_dims, n_canvas_channels=dec_canvas_chans, output_type=DistributionLayer.Types.NORMAL_UNITVAR),
        latent_dim=latent_dims
        ),
    )  # should last about 3.5 hours


# Suprisingly, this does not drop latent the last N-2 latent dimensions!
X_vae_supervised = X_vae.add_variant(supervision_schedule = 1.)

# To enforce the drop of latent dimensions, we disable reparametrization trick
X_vae_supervised_separate = X_vae_supervised.add_variant(z_sample_schedule = 'no_reparametrization')

X_vae_supervised_separate_zeroed = X_vae_supervised_separate.add_variant(zero_irrelevant_latents_schedule = True)

# It seems that X_vae and X_vae_supervised and do about equally well, and that surprisingly, X_vae_supervised_separate
# and X_vae_supervised_separate_zeroed basically fail, regardless of the zero_init_irrelevant_latents setting.

# X_vae_supervised_target = X_vae.add_variant('vae_supervised_target', z_sample_schedule = 'target', supervision_schedule = 1., zero_init_irrelevant_latents = True)
# X_vae_supervised_target_init = X_vae.add_variant('vae_supervised_target_init', z_sample_schedule = {0: 'target', 5000: 'natural'}, supervision_schedule = {0: 1., 5000: 0.}, zero_init_irrelevant_latents = True)


X_vae_designed = X_vae_supervised.add_variant('designed', zero_irrelevant_latents_schedule = {0: True, 10000: False}, supervision_schedule = {0: 1, 10000: 0}, z_sample_schedule={0: 'target', 10000: 'natural'})
X_vae_designed_kl = X_vae_supervised.add_variant('designed_kl', zero_irrelevant_latents_schedule = {0: True, 10000: False}, supervision_schedule = {0: 1, 10000: 0}, z_sample_schedule={0: 'target_kl', 10000: 'natural'})


# X_vae_initially_supervised = X_vae.add_variant(supervision_schedule = {0: 1, 10000: 0}, z_sample_schedule={0: 'target', 10000: None})

if __name__ == '__main__':
    print('target_kl')
    X_vae.browse()
    # X_vae_supervised.call()
    # X_vae_supervised_separate.call()R
    # X_vae_initially_supervised.call()
    # X_vae_supervised_separate_zeroed.call()
