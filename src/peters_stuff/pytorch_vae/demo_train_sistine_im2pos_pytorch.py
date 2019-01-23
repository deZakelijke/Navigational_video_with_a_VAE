from __future__ import print_function

import random
import string

import numpy as np
import time
import torch
from torch.optim import Adam
from typing import Callable, Tuple, Iterator

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from src.peters_stuff.bbox_utils import position_to_bbox
from src.peters_stuff.image_crop_generator import batch_crop, \
    iter_bbox_batches
from src.peters_stuff.pytorch_vae.convlstm import ConvLSTMImageToPositionEncoder
from src.peters_stuff.pytorch_vae.interfaces import IImageToPositionEncoder
from src.peters_stuff.pytorch_vae.pytorch_helpers import setup_cuda_if_available
from src.peters_stuff.pytorch_vae.pytorch_imutils import get_normed_crops_and_position_tensors
from src.peters_stuff.sample_data import SampleImages


def chanfirst(im):
    return np.rollaxis(im, 3, 1)


def chanlast(im):
    return np.rollaxis(im, 1, 4)


def normalize_image(im):
    return torch.from_numpy((chanfirst(im)/127.)-1.).float()


def denormalize_image(im):
    return ((chanlast(im.detach().cpu().numpy())+1.)*127.)



def generate_random_model_path(code_gen_len=16, suffix='.pth'):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(code_gen_len))
    model_path = get_artemis_data_path('models/{}{}'.format(code, suffix), make_local_dir=True)
    return model_path



@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='pixel_error'), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_convlstm_pos2im(
        model = IImageToPositionEncoder,
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

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)

    save_path = generate_random_model_path()

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor, n_iter=n_iter)):

        raw_image_crops, normed_image_crops, positions = get_normed_crops_and_position_tensors(img=img, bboxes=bboxes)

        predicted_position_dist = model(normed_image_crops)

        loss = ((predicted_position_dist.mean - positions)**2).flatten(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_positions = predicted_position_dist.mean
        error = abs(predicted_positions - positions).mean().item()
        duck[next, :] = dict(iter=i, rel_error=error, elapsed=time.time()-t_start, training_loss=loss.item())
        if do_every('20s'):
            report = f'Iter: {i}, Rel Error: {error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)
            with hold_dbplots():
                positions = positions.detach().cpu().numpy()
                predicted_positions = predicted_positions.detach().cpu().numpy()
                dbplot(raw_image_crops[..., :3], 'crops')
                predicted_bboxes = position_to_bbox(predicted_positions, image_size=img.shape[:2], crop_size=crop_size, clip=True)
                dbplot(batch_crop(img=img, bboxes=predicted_bboxes), 'predicted_crops')
                # if append_position_maps:
                #     dbplot(np.rollaxis(image_crops[0, :, :, 3:], 2, 0), 'posmaps')
                dbplot(duck[:, 'rel_error'].to_array(), plot_type=DBPlotTypes.LINE)
                dbplot((positions[:, 0], positions[:, 1]), 'positions', plot_type=DBPlotTypes.SCATTER)
                dbplot((predicted_positions[:, 0], predicted_positions[:, 1]), 'predicted_positions', plot_type=DBPlotTypes.SCATTER)
                # dbplot(predicted_imgs, 'predicted_crops', cornertext=report)

        if is_checkpoint():
            save_figure_in_record()
            if save_models:
                torch.save(model, save_path)
                print(f'Model saved to {save_path}')
            yield duck


    # for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor, n_iter=n_iter)):
    #
    #
    #     image_crops = normalize_image(batch_crop(img=img, bboxes=bboxes)).to(get_default_device()).float()
    #
    #     positions = torch.from_numpy(bbox_to_position(bboxes=bboxes, image_size=img.shape[:2], crop_size=crop_size)).to(get_default_device()).float()
    #
    #     predicted_dist = model(positions)
    #
    #     # loss = -predicted_dist.log_prob(image_crops).flatten(1).sum(dim=1).mean()
    #     # loss = ((predicted_dist.mean - image_crops)**2).flatten(1).sum(dim=1).mean()
    #     loss = ((predicted_dist.mean - image_crops)**2).flatten(1).mean()
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     true_image = denormalize_image(image_crops[:16])
    #     predicted_image = denormalize_image(predicted_dist.mean[:16])
    #
    #     pixel_error = np.abs(true_image - predicted_image).mean()/255.
    #
    #     duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=loss.item())
    #
    #     # if do_every('30s'):
    #     if do_every(100):
    #         report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
    #         print(report)
    #
    #         with hold_dbplots():
    #             dbplot(true_image, 'crops')
    #             dbplot(predicted_image, 'predicted_crops', cornertext=report)
    #
    #     if is_checkpoint():
    #         save_figure_in_record()
    #         if save_models:
    #             torch.save(model, save_path)
    #             print(f'Model saved to {save_path}')
    #         yield duck


Xgqn3=demo_train_convlstm_pos2im.add_root_variant(save_models=True).add_config_variant('gqn3',
   model = lambda n_hidden_channels=32:
    ConvLSTMImageToPositionEncoder(input_shape=(3, 64, 64), n_hidden_channels=n_hidden_channels))


if __name__ == '__main__':

    # Xgqn3.run()

    # Xgqn3.browse()
    Xgqn3.call()

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
