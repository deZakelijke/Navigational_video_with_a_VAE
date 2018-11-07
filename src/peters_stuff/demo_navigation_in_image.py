import time
from collections import namedtuple
from typing import NamedTuple, Callable, Iterator, Tuple, Union

import numpy as np
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from artemis.fileman.file_getter import get_file
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.duck import Duck
from artemis.general.image_ops import resize_image
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from src.peters_stuff.crop_predictors import CropPredictorNodes, ICropPredictor
from src.peters_stuff.image_crop_generator import iter_bbox_batches, batch_crop
from src.peters_stuff.position_predictors import PositionPredictorNodes, IPositionPredictor, bbox_to_position, \
    position_to_bbox
from src.peters_stuff.sample_data import SampleImages
from src.peters_stuff.tf_helpers import TFGraphClass
import tensorflow as tf


class INavigationModel(object):

    def plan_route(self, start_img, dest_img):
        raise NotImplementedError()




class PretrainedStrightLineNavModel(INavigationModel):

    def __init__(self, encoder: IPositionPredictor, decoder: ICropPredictor, n_waypoints):
        self.encoder = encoder
        self.decoder = decoder
        self.frac = np.linspace(0, 1, n_waypoints)[:, None]

    def plan_route(self, start_img, dest_img):
        route, zs = self.plan_route_and_get_zs(start_img, dest_img)

    def plan_route_and_get_zs(self, start_img, dest_img):
        zs = self.encoder.predict(start_img[None])
        zd = self.encoder.predict(dest_img[None])
        # dbplot(self.decoder.predict(zs), 'start_recon')
        # dbplot(self.decoder.predict(zd), 'dest_recon')
        zp = zs*(1-self.frac) + zd*self.frac
        print(f'zs={zs}, zd={zd}')
        video = self.decoder.predict(zp)
        return video, zp

# class EncDecNodes(NamedTuple):
#     enc: PositionPredictorNodes
#     dec: CropPredictorNodes
#
#
# class PretrainedDifferentiableGraphLoader(TFGraphClass, INavigationModel):
#
#     def plan_route(self, start_img, dest_img):
#         raise NotImplementedError()
#
#
#     @staticmethod
#     def from_pretrained_encdec(encoder_path, decoder_path, image_size, n_waypoints = 100):
#         encoder_nodes = TFGraphClass.load(encoder_path, scope = 'enc').nodes  # type: PositionPredictorNodes
#         decoder_nodes = TFGraphClass.load(decoder_path, scope = 'dec').nodes  # type: CropPredictorNodes
#
#
#
#         frac = tf.linspace(0, 1, n_waypoints)[:, None]
#
#
#
#         batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Yes it's silly but we have to
#         z_start = tf.placeholder(dtype=tf.float32, shape=(1, *image_size, 3))
#         z_dest = tf.placeholder(dtype=tf.float32, shape=(1, *image_size, 3))
#
#         loss = tf.abs(decoder_nodes.predicted_crops[1:]-decoder_nodes.predicted_crops[:-1]).sum()
#         opt = GradientDescentOptimizer(learning_rate=0.01)
#         update_up = opt.minimize(loss)
#
#         nodes = EncDecNodes(enc=encoder_nodes, dec=decoder_nodes)
#         return PretrainedDifferentiableGraphLoader(nodes=nodes)


def demo_show_navigation_video(
        model,
        position_generator_constructor: Union[str, Callable[[], Iterator[Tuple[int, int]]]] = 'random',
        crop_size = (64, 64),
        ):

    img = SampleImages.sistine_512()
    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=2, position_generator_constructor=position_generator_constructor)):
        start_image, dest_image = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))
        route_video, zs = model.plan_route_and_get_zs(start_image, dest_image)
        true_positions = bbox_to_position(bboxes, image_size=img.shape[:2], crop_size=start_image.shape[:2])
        for p_true, p_enc in zip(true_positions, zs[[0, -1]]):
            print(f'{p_true} -> {p_enc}')
        with hold_dbplots():
            dbplot(start_image, 'start')
            dbplot(dest_image, 'dest')
            dbplot(img, 'full_img')
        for t, im in enumerate(route_video):
            dbplot(im, 'route', title=f'Route - Frame {t+1}/{len(route_video)}', hang=1. if t in (0, len(route_video)-1) else None)



class CrossCorrelationController(object):

    def __init__(self):
        pass

    def __call__(self, current_frame, next_frame):
        pass

def demo_use_system_controller(
        model,
        position_generator_constructor: Union[str, Callable[[], Iterator[Tuple[int, int]]]] = 'random',
        crop_size = (64, 64),
        ):

    img = SampleImages.sistine_512()
    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=crop_size, batch_size=2, position_generator_constructor=position_generator_constructor)):
        start_image, dest_image = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))
        route_video, zs = model.plan_route_and_get_zs(start_image, dest_image)
        true_positions = bbox_to_position(bboxes, image_size=img.shape[:2], crop_size=start_image.shape[:2])
        for p_true, p_enc in zip(true_positions, zs[[0, -1]]):
            print(f'{p_true} -> {p_enc}')
        with hold_dbplots():
            dbplot(start_image, 'start')
            dbplot(dest_image, 'dest')
            dbplot(img, 'full_img')
        for t, im in enumerate(route_video):
            dbplot(im, 'route', title=f'Route - Frame {t+1}/{len(route_video)}', hang=1. if t in (0, len(route_video)-1) else None)




if __name__ == "__main__":

    encoder = TFGraphClass.load(get_artemis_data_path('tests/models/ROTBKQWX3DHT30D2/model'))
    decoder = TFGraphClass.load(get_artemis_data_path('tests/models/IIUGUDCOADWOABMB/model'))
    demo_train_just_vae_on_images_gqn(model=PretrainedStrightLineNavModel(encoder=encoder, decoder=decoder, n_waypoints=64))

