from __future__ import print_function

import numpy as np
from artemis.experiments.decorators import ExperimentFunction

from artemis.experiments.experiment_record import save_figure_in_record

from artemis.general.should_be_builtins import bad_value

from artemis.general.duck import Duck

from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.measuring_periods import measure_period
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.image_crop_generator import batch_crop, iter_bbox_batches
from src.peters_stuff.sample_data import SampleImages
from src.peters_stuff.sweeps import generate_linear_sweeps
from src.peters_stuff.tf_vaes import TFVAEModel, get_convlstm_vae_graph


@ExperimentFunction(one_liner_function=lambda result: f'Loss: {result[-len(result)//10:, "loss"].to_array().mean():.3g}')
def demo_train_just_vae_on_images(
        model_constructor = 'convlstm_1',
        batch_size=64,
        checkpoints={0:10, 100:100, 1000: 400},
        latent_dims = 2,
        kl_scale = 1.,
        crop_size = (64, 64),
        n_iter = None,
        output_type = 'bernoulli',
        seed=1234,
        data_seed = 1235,
        save_model = False,

        ):

    if model_constructor == 'convlstm_1':
        model = TFVAEModel(get_convlstm_vae_graph(n_pose_channels=latent_dims, output_type = output_type, kl_scale=kl_scale))
    elif model_constructor == 'convlstm_2':
        model = TFVAEModel(get_convlstm_vae_graph(n_pose_channels=latent_dims, output_type = output_type, kl_scale=kl_scale, n_gen_maps=128))

    else:
        assert callable(model_constructor)
        model = model_constructor()
    data_rng = np.random.RandomState(data_seed)

    is_checkpoint = Checkpoints(checkpoints)

    img = SampleImages.sistine_512()
    # img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')
    # img = img[img.shape[0]//2-image_cut_size[0]//2:img.shape[0]//2+image_cut_size[0]//2, img.shape[1]//2-image_cut_size[1]//2:img.shape[1]//2+image_cut_size[1]//2]  # TODO: Revert... this is just to test on a smaller version

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
            if save_model:
                save_path = model.dump()
                print(f'Model saved to {save_path}')

            yield data


X_trial = demo_train_just_vae_on_images.add_root_variant(n_iter=10000)

X_trial.add_variant(output_type='normal', latent_dims=20, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=20, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=3, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=2, kl_scale=1.)
X_trial.add_variant(output_type='bernoulli', latent_dims=20, kl_scale=10.)

X_long = demo_train_just_vae_on_images.add_variant('long', model_constructor='convlstm_2', output_type='bernoulli', latent_dims=20, kl_scale=1., save_model=True)

# X_long_XXX = demo_train_just_vae_on_images.add_root_variant(save_model=True).add_config_variant('convlstm',
#     model_constructor = lambda latent_dims, output_types, n_pose_channels=20, n_gen_maps=128, n_rec_maps=32, the_output_type='bernoulli':
#         lambda: TFVAEModel(get_convlstm_vae_graph(output_type=output_types, n_pose_channels=latent_dims, n_gen_maps=n_gen_maps, n_rec_maps=n_rec_maps, image_shape=(64, 64, 3)))
#     )  # Need to fix artemis so that this works.


if __name__ == '__main__':
    # demo_train_just_vae_on_images.get_variant('search').run()
    # demo_train_just_vae_on_images.browse()
    X_long.run()
