from __future__ import print_function

import pickle

import numpy as np
import time
from typing import Callable, Tuple, Iterator
import os
from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import get_current_record_dir, save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.general.image_ops import resize_image
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.crop_predictors import ICropPredictor
from src.peters_stuff.crop_predictors_torch import DeconvCropPredictor
from src.peters_stuff.crop_predictors_tf import GQNCropPredictor, GQNCropPredictor2, GQNCropPredictor3
from src.peters_stuff.image_crop_generator import iter_bboxes_from_positions, iter_pos_random, batch_crop, \
    iter_bbox_batches
from src.peters_stuff.position_predictors import bbox_to_position
from src.peters_stuff.sample_data import SampleImages


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='pixel_error'), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_just_vae_on_images_gqn(
        model_constructor: Callable[[int, Tuple[int,int]], ICropPredictor],
        position_generator_constructor: Callable[[], Iterator[Tuple[int, int]]] = 'random',
        batch_size=64,
        checkpoints={0:100, 1000: 1000},
        crop_size = (64, 64),
        n_iter = None,
        save_models = False,
        ):

    img = SampleImages.sistine_512()

    dbplot(img, 'full_img')
    model = model_constructor(batch_size, crop_size)

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)
    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor)):

        if n_iter is not None and i>=n_iter:
            break

        image_crops = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))

        positions = bbox_to_position(bboxes=bboxes, image_size=img.shape[:2], crop_size=crop_size)

        predicted_imgs, training_loss = model.train(positions, image_crops)

        pixel_error = np.abs(predicted_imgs - image_crops).mean()/255.

        duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=training_loss)

        if do_every('30s'):
            report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)
            with hold_dbplots():
                dbplot(image_crops[:16], 'crops')
                dbplot(predicted_imgs[:16], 'predicted_crops', cornertext=report)

        if is_checkpoint():
            save_figure_in_record()
            if save_models:
                save_path = model.dump()
                print(f'Model saved to {save_path}')
            yield duck


X = demo_train_just_vae_on_images_gqn.add_config_root_variant('deconv1', model_constructor = DeconvCropPredictor.get_constructor)
X32 = X.add_variant(filters = 32, n_iter=10000)
X64 = X.add_variant(filters = 64, n_iter=10000)
X128 = X.add_variant(filters = 128, n_iter=10000)
X256 = X.add_variant(filters = 256, n_iter=10000)

Xgqn=demo_train_just_vae_on_images_gqn.add_config_variant('gqn1', model_constructor = GQNCropPredictor.get_constructor)
Xgqn2=demo_train_just_vae_on_images_gqn.add_config_variant('gqn2', model_constructor = GQNCropPredictor2.get_constructor)

Xgqn2.add_variant(n_maps=64, canvas_channels=64)
Xgqn2.add_variant(n_maps=64, canvas_channels=32)
Xgqn2.add_variant(n_maps=32, canvas_channels=32)
Xgqn2.add_variant(n_maps=64, canvas_channels=64, sequence_size=6)

Xgqn3=demo_train_just_vae_on_images_gqn.add_root_variant(save_models=True).add_config_variant('gqn3', model_constructor = GQNCropPredictor3.get_constructor)
Xgqn3.add_variant('params', n_maps=128, canvas_channels=64)

if __name__ == '__main__':

    # Xgqn2.get_variant(n_maps=64, canvas_channels=64).call()
    # Xgqn3.run()

    # Xgqn.call()
    # Xgqn2.run()
    # Xgqn2_params.call()
    # X64.run()
    # X32.run()
    # X64.run()
    # X128.run()
    # demo_train_just_vae_on_images_gqn()
    demo_train_just_vae_on_images_gqn.browse(raise_display_errors=True)

    # demo_train_just_vae_on_images_gqn.get_variant('deconv1').run()
    # demo_train_just_vae_on_images_gqn.get_variant('gqn1').call()
