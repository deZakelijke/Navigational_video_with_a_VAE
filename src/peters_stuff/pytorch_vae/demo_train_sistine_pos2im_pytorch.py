from __future__ import print_function

import time
from typing import Callable, Tuple, Iterator

import numpy as np
import torch
from torch.optim import Adagrad, Adam

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record import save_figure_in_record
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.peters_stuff.crop_predictors import ICropPredictor
from src.peters_stuff.image_crop_generator import batch_crop, \
    iter_bbox_batches
from src.peters_stuff.position_predictors import bbox_to_position
from src.peters_stuff.pytorch_helpers import get_default_device, set_default_device
from src.peters_stuff.pytorch_vae.convlstm import ConvLSTMPositiontoImageDecoder
from src.peters_stuff.sample_data import SampleImages
import random
import string


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
        model = ICropPredictor,
        position_generator_constructor: Callable[[], Iterator[Tuple[int, int]]] = 'random',
        batch_size=64,
        checkpoints={0:100, 1000: 1000},
        crop_size = (64, 64),
        n_iter = None,
        save_models = False,
        ):


    img = SampleImages.sistine_512()

    cuda = torch.cuda.is_available()
    if cuda:
        model.to('cuda')
        torch.set_default_tensor_type(torch.FloatTensor)
        set_default_device('cuda')
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f'Cuda: {cuda}')

    optimizer = Adagrad(lr=1e-3, params = model.parameters())
    # optimizer = Adam(params = model.parameters())

    dbplot(img, 'full_img')
    # model = model_constructor(batch_size, crop_size)

    duck = Duck()
    t_start = time.time()
    is_checkpoint = Checkpoints(checkpoints)

    save_path = generate_random_model_path()

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=batch_size, position_generator_constructor=position_generator_constructor)):

        if n_iter is not None and i>=n_iter:
            break

        image_crops = normalize_image(batch_crop(img=img, bboxes=bboxes)).to(get_default_device()).float()

        positions = torch.from_numpy(bbox_to_position(bboxes=bboxes, image_size=img.shape[:2], crop_size=crop_size)).to(get_default_device()).float()

        predicted_dist = model(positions)

        # loss = -predicted_dist.log_prob(image_crops).flatten(1).sum(dim=1).mean()
        loss = ((predicted_dist.mean - image_crops)**2).flatten(1).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pixel_error = torch.abs(image_crops - predicted_dist.mean).mean().item()

        duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=loss.item())

        if do_every('30s'):
            report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)

            with hold_dbplots():
                dbplot(denormalize_image(image_crops[:16]), 'crops')
                dbplot(denormalize_image(predicted_dist.mean[:16]), 'predicted_crops', cornertext=report)

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
