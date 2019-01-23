from __future__ import print_function

import numpy as np
import time
import torch
from torch.optim import Adam
from typing import Callable, Union

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.image_crop_generator import iter_bbox_batches
from src.peters_stuff.pytorch_vae.convlstm import ConvLSTMPositiontoImageDecoder
from src.peters_stuff.pytorch_vae.interfaces import IPositionToImageDecoder
from src.peters_stuff.pytorch_vae.pytorch_helpers import setup_cuda_if_available
from src.peters_stuff.pytorch_vae.pytorch_imutils import denormalize_image, generate_random_model_path, \
    get_normed_crops_and_position_tensors
from src.peters_stuff.sample_data import SampleImages


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='pixel_error'), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_convlstm_pos2im(
        model: IPositionToImageDecoder,
        position_generator_constructor: Union[str, Callable] = 'random',
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

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)

    save_path = generate_random_model_path()

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor, n_iter=n_iter)):

        raw_image_crops, normed_image_crops, positions = get_normed_crops_and_position_tensors(img=img, bboxes=bboxes)

        predicted_dist = model(positions)

        loss = ((predicted_dist.mean - normed_image_crops)**2).flatten(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_image = denormalize_image(predicted_dist.mean[:16])
        pixel_error = np.abs(raw_image_crops[:16] - predicted_image).mean()/255.

        duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=loss.item())

        if do_every(100):
            report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)

            with hold_dbplots():
                dbplot(raw_image_crops, 'crops')
                dbplot(predicted_image, 'predicted_crops', cornertext=report)

        if is_checkpoint():
            save_figure_in_record()
            if save_models:
                torch.save(model, save_path)
                print(f'Model saved to {save_path}')
            yield duck


Xgqn3=demo_train_convlstm_pos2im.add_root_variant(save_models=True).add_config_variant('gqn3',
   model = lambda n_hidden_channels=128, n_canvas_channels=64:
    ConvLSTMPositiontoImageDecoder(input_shape=(3, 64, 64), n_hidden_channels=n_hidden_channels, n_canvas_channels=n_canvas_channels))
# Xgqn3.add_variant('params', n_maps=128, canvas_channels=64)

if __name__ == '__main__':

    # Xgqn3.run()

    Xgqn3.browse()
    # Xgqn3.call()

    # Xgqn.call()
    # Xgqn2.run()
    # Xgqn2_params.call()
    # X64.run()
    # X32.run()
    # X64.run()
    # X128.run()
    # demo_train_just_vae_on_images_gqn()
    # demo_train_convlstm_pos2im.browse(raise_display_errors=True)
    # demo_train_convlstm_pos2im.browse(raise_display_errors=True)

    # demo_train_just_vae_on_images_gqn.get_variant('deconv1').run()
    # demo_train_just_vae_on_images_gqn.get_variant('gqn1').call()
