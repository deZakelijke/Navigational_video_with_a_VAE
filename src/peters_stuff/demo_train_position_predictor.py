from __future__ import print_function

from abc import abstractmethod

import numpy as np
import tensorflow as tf
import time
from tensorflow.python.training.adam import AdamOptimizer
from torch.nn.functional import binary_cross_entropy
from typing import Callable, Tuple, Iterator

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    get_timeseries_oneliner_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.general.image_ops import resize_image
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from src.gqn.gqn_draw import generator_rnn
from src.peters_stuff.image_crop_generator import iter_bboxes_from_positions, iter_pos_random, batch_crop
from src.peters_stuff.position_predictors import IPositionPredictor, feat_to_imbatch, ConvnetPositionPredictor, \
    ConvnetGridPredictor, ConvnetGridPredictor2, bbox_to_position, position_to_bbox, GQNPositionPredictor, \
    GQNPositionPredictor2, GQNPositionPredictor3


#
# class GQNCropPredictor(IPositionPredictor):
#
#     def __init__(self, batch_size, image_size):
#         set_gqn_param('POSE_CHANNELS', 2)
#         enc_h, enc_w = get_gqn_param('ENC_HEIGHT'), get_gqn_param('ENC_WIDTH')
#         g = Namespace()
#         g.positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
#         g.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
#         g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, enc_h, enc_w, 1))
#         g.mu_targ, _ = generator_rnn(representations=g.representations, query_poses=g.positions, sequence_size=12)
#         g.loss = tf.reduce_mean((g.mu_targ-g.targets)**2)
#         g.update_op = AdamOptimizer().minimize(g.loss)
#         sess = tf.Session()
#         sess.run(tf.global_variables_initializer())
#         self.g = g
#         self.sess = sess
#
#     def train(self, image_crops, positions, ):
#         image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
#         predicted_imgs, _, loss = self.sess.run([self.g.mu_targ, self.g.update_op, self.g.loss] , feed_dict={self.g.positions: positions, self.g.targets: image_crops})
#
#         # with hold_dbplots(draw_every=10):  # Just check that data normalization is right
#         #     dbplot(predicted_imgs, 'preed')
#         #     dbplot(image_crops, 'crooops')
#         return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss
#
#     def predict(self, positions):
#         predicted_imgs, _, loss = self.sess.run([self.g.mu_targ] , feed_dict={self.g.positions: positions})
#         return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss
#
#     @staticmethod
#     def get_constructor():
#         return lambda batch_size, image_size: GQNCropPredictor(batch_size=batch_size, image_size=image_size)
from src.peters_stuff.sample_data import SampleImages


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='rel_error'), one_liner_function=lambda result: f'iter:{result[-1]["iter"]}, rel_error:[min:{min(result[:, "rel_error"]):.3g}, final:{result[-1, "rel_error"]:.3g} ]')
def demo_train_position_predictor(
        model_constructor: Callable[[int, Tuple[int,int]], IPositionPredictor],
        position_generator_constructor: Callable[[], Iterator[Tuple[int, int]]] = 'default',
        batch_size=64,
        checkpoints='60s',
        crop_size = (64, 64),
        # image_cut_size = (128, 128),  # (y, x)
        image_cut_size = (512, 512),  # (y, x)
        n_iter = 10000,
        append_position_maps = False,
        save_model=False,
        ):

    if isinstance(position_generator_constructor, str):
        if position_generator_constructor=='default':
            position_generator_constructor = lambda: iter_pos_random(n_dim=2, rng=None)
        else:
            raise NotImplementedError(position_generator_constructor)

    is_checkpoint = Checkpoints(checkpoints)

    img = SampleImages.sistine_512()
    dbplot(img, 'full_img')

    model = model_constructor(batch_size, crop_size)

    duck = Duck()
    batched_bbox_generator = batchify_generator(list(
        iter_bboxes_from_positions(
            img_size=img.shape[:2],
            crop_size=crop_size,
            position_generator=position_generator_constructor(),
        ) for _ in range(batch_size)))

    if append_position_maps:
        position_maps = np.broadcast_to(
            feat_to_imbatch(np.array(np.meshgrid(*(np.linspace(0, 1, cs) for cs in crop_size)))[None], channel_first=True, datarange=(0, 1)), (batch_size, *crop_size, 2))

    t_start = time.time()
    for i, bboxes in enumerate(batched_bbox_generator):
        if n_iter is not None and i>=n_iter:
            break

        image_crops = batch_crop(img=img, bboxes=bboxes)

        if append_position_maps:
            image_crops = np.concatenate([image_crops, position_maps], axis=-1)

        positions = bbox_to_position(bboxes, image_size=img.shape[:2], crop_size=crop_size)
        predicted_positions, training_loss = model.train(image_crops, positions)
        error = np.abs(predicted_positions - positions).mean()
        duck[next, :] = dict(iter=i, rel_error=error, elapsed=time.time()-t_start, training_loss=training_loss)
        if do_every('20s'):
            report = f'Iter: {i}, Rel Error: {error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)
            with hold_dbplots():
                dbplot(image_crops[..., :3], 'crops')
                predicted_bboxes = position_to_bbox(predicted_positions, image_size=img.shape[:2], crop_size=crop_size, clip=True)
                dbplot(batch_crop(img=img, bboxes=predicted_bboxes), 'predicted_crops')
                if append_position_maps:
                    dbplot(np.rollaxis(image_crops[0, :, :, 3:], 2, 0), 'posmaps')
                dbplot(duck[:, 'rel_error'].to_array(), plot_type=DBPlotTypes.LINE)
                dbplot((positions[:, 0], positions[:, 1]), 'positions', plot_type=DBPlotTypes.SCATTER)
                dbplot((predicted_positions[:, 0], predicted_positions[:, 1]), 'predicted_positions', plot_type=DBPlotTypes.SCATTER)
                # dbplot(predicted_imgs, 'predicted_crops', cornertext=report)
        if is_checkpoint():
            if save_model:
                save_path = model.dump()
                print(f'Model saved to {save_path}')
            yield duck

    yield duck


X = demo_train_position_predictor.add_config_root_variant('deconv1', model_constructor = ConvnetPositionPredictor.get_constructor)
X32 = X.add_variant(filters = 32)
X32.add_variant(opt='sgd')
X32a = X32.add_variant(img_channels=5, append_position_maps=True)
X32b = X32.add_variant(use_batchnorm=False)
X64 = X.add_variant(filters = 64)
# X64a = X32.add_variant(img_channels=5, append_position_maps=True)
# X64 = X.add_variant(filters = 64, n_iter=10000)
X128 = X.add_variant(filters = 128)
# X256 = X.add_variant(filters = 256, n_iter=10000)

# demo_train_just_vae_on_images_gqn.add_config_variant('gqn1', model_constructor = GQNCropPredictor.get_constructor)

Xgrid = demo_train_position_predictor.add_config_root_variant('convgrid', model_constructor = ConvnetGridPredictor.get_constructor)
Xgrid.add_variant(opt='adam')
Xgridsgd = Xgrid.add_variant(opt='sgd')
Xgrid.add_variant(opt='sgd-mom')
Xgrid.add_variant(opt='rmsprop')
Xgridsgd.add_variant('posmaps', img_channels=5, append_position_maps=True)
Xgridsgd.add_variant(learning_rate = 1e-2)
Xgridsgd.add_variant(gridsize=(20, 20))
Xgridsgd.add_variant(gridsize=(5, 5))
Xgridsgd.add_variant(gridsize=(3, 3))
Xgridsgd.add_variant(learning_rate = 1e-4)
Xgridsgd.add_variant(use_batchnorm = False)
Xgridsgd.add_variant(filters = 64)


XV2 = demo_train_position_predictor.add_config_root_variant('convgrid2', model_constructor = ConvnetGridPredictor2.get_constructor)
Xgqn = demo_train_position_predictor.add_config_root_variant('gqn', model_constructor = GQNPositionPredictor.get_constructor)
Xgqn2 = demo_train_position_predictor.add_config_root_variant('gqn2', model_constructor = GQNPositionPredictor2.get_constructor)
# Xgqn_params = Xgqn.add_variant(rnn_params = {})
# Xgqn_params = Xgqn.add_variant(enc_h=4, enc_w = 4, rnn_params = dict(lstm_canvas_channels=64))
# Xgqn_params = Xgqn.add_variant(enc_h=4, enc_w = 4, )


Xgqn_THIS_WORKS_DONT_TOUCH = Xgqn.add_variant(rnn_params = dict(lstm_canvas_channels=0, lstm_output_channels=64, generator_input_channels=0, inference_input_channels=0))
# Xgqn_params = Xgqn.add_variant(n_maps=32)
Xgqn3 = demo_train_position_predictor.add_root_variant(save_model=True).add_config_root_variant('gqn3', model_constructor = GQNPositionPredictor3.get_constructor)

if __name__ == '__main__':
    # Xgqn_params.run()
    # Xgqn_THIS_WORKS_DONT_TOUCH.call()
    # Xgqn2.call()
    Xgqn3.run()
    # X64.run()
    # X32.call(learning_rate=1e-3)
    # Xgrid.call(n_iter=100000)
    # X32.call()
    # Xgrid.call()
    # X32a.run()
    # X64.run()
    # X128.run()
    # demo_train_just_vae_on_images_gqn()
    # demo_train_position_predictor.browse()
    # XV2.call(opt='sgd', learning_rate = 1e-3, weight_decay=0.0001)
    # Xgridsgd.get_variant(gridsize=(5, 5)).call()

    # Xgqn.call()

    # Xgridsgd.get_variant(gridsize=(5, 5)).call()
    #
    # demo_train_just_vae_on_images_gqn.get_variant('deconv1').run()
    # demo_train_just_vae_on_images_gqn.get_variant('gqn1').call()
