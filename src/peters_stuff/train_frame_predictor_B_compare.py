from __future__ import print_function

from abc import abstractmethod

import numpy as np
import tensorflow as tf
import time
from argparse import Namespace
from tensorflow.python.training.adam import AdamOptimizer
from torch.nn.functional import binary_cross_entropy
from typing import Callable, Tuple, Iterator

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    timeseries_oneliner_function, get_timeseries_oneliner_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.duck import Duck
from artemis.general.image_ops import resize_image
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from src.VAE_with_Disc import VAE
from src.gqn.gqn_draw import generator_rnn
from src.gqn.gqn_params import set_gqn_param, get_gqn_param
from src.peters_stuff.image_crop_generator import iter_bboxes_from_positions, iter_pos_random, batch_crop


class ICropPredictor(object):

    @abstractmethod
    def train(self, positions, image_crops):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, positions):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_constructor(**kwargs) -> Callable[[int, Tuple[int,int]], 'ICropPredictor']:
        raise NotImplementedError()


def imbatch_to_feat(im, channel_first, datarange):

    if channel_first:
        im = np.rollaxis(im, 3, 1)
    dmin, dmax = datarange
    feat = (im.astype(np.float32)/255.999)*(dmax-dmin)+dmin
    return feat


def feat_to_imbatch(feat, channel_first, datarange):
    if channel_first:
        feat = np.rollaxis(feat, 1, 4)
    dmin, dmax = datarange
    feat = (((feat-dmin)/(dmax-dmin))*255.999).astype(np.uint8)
    return feat


class GQNCropPredictor(ICropPredictor):

    def __init__(self, batch_size, image_size):
        set_gqn_param('POSE_CHANNELS', 2)
        enc_h, enc_w = get_gqn_param('ENC_HEIGHT'), get_gqn_param('ENC_WIDTH')
        g = Namespace()
        g.positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
        g.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
        g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, enc_h, enc_w, 1))
        g.mu_targ, _ = generator_rnn(representations=g.representations, query_poses=g.positions, sequence_size=12)
        g.loss = tf.reduce_mean((g.mu_targ-g.targets)**2)
        g.update_op = AdamOptimizer().minimize(g.loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.g = g
        self.sess = sess

    def train(self, positions, image_crops):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_imgs, _, loss = self.sess.run([self.g.mu_targ, self.g.update_op, self.g.loss] , feed_dict={self.g.positions: positions, self.g.targets: image_crops})

        # with hold_dbplots(draw_every=10):  # Just check that data normalization is right
        #     dbplot(predicted_imgs, 'preed')
        #     dbplot(image_crops, 'crooops')
        return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss

    def predict(self, positions):
        predicted_imgs, _, loss = self.sess.run([self.g.mu_targ] , feed_dict={self.g.positions: positions})
        return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss

    @staticmethod
    def get_constructor():
        return lambda batch_size, image_size: GQNCropPredictor(batch_size=batch_size, image_size=image_size)


class DeconvCropPredictor(ICropPredictor):

    def __init__(self, image_size, learning_rate=1e-3, filters=32, loss_type='mse'):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VAE(latent_dims=2, image_size=image_size, filters=filters).to(self.device)
        self.opt = torch.optim.Adam(list(self.model.parameters()), lr = learning_rate, betas = (0.5, 0.999))
        self.loss_type = loss_type

    def train(self, positions, image_crops):
        import torch
        predicted_imgs = self.model.decode(torch.Tensor(positions).to(self.device))
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)

        # with hold_dbplots(draw_every=10):  # Just check that data normalization is right
        #     dbplot(predicted_imgs, 'preed', plot_type=DBPlotTypes.CIMG)
        #     dbplot(var_image_crops, 'crooops', plot_type=DBPlotTypes.CIMG)
        if self.loss_type=='bce':
            loss = torch.nn.functional.binary_cross_entropy(predicted_imgs, var_image_crops, size_average = False)
        elif self.loss_type=='mse':
            loss = torch.nn.functional.mse_loss(predicted_imgs, var_image_crops, size_average = True)
        else:
            raise NotImplementedError(self.loss_type)
        loss.backward()
        self.opt.step()
        return feat_to_imbatch(predicted_imgs.detach().cpu().numpy(), channel_first=True, datarange=(0, 1)), loss.detach().cpu().numpy()

    def predict(self, positions):
        import torch
        predicted_imgs = self.model.decode(torch.Tensor(positions).to(self.device))
        return feat_to_imbatch(predicted_imgs.detach().cpu().numpy(), channel_first=True, datarange=(0, 1))

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32):
        return lambda batch_size, image_size: DeconvCropPredictor(image_size=image_size, learning_rate=learning_rate, filters=filters)


@ExperimentFunction(is_root=True, compare=get_timeseries_record_comparison_function(yfield='pixel_error'), one_liner_function=get_timeseries_oneliner_function(fields = ['iter', 'pixel_error']))
def demo_train_just_vae_on_images_gqn(
        model_constructor: Callable[[int, Tuple[int,int]], ICropPredictor],
        position_generator_constructor: Callable[[], Iterator[Tuple[int, int]]] = 'default',
        batch_size=64,
        checkpoints={0:10, 100:100, 1000: 1000},
        crop_size = (64, 64),
        # image_cut_size = (128, 128),  # (y, x)
        image_cut_size = (512, 512),  # (y, x)
        n_iter = None,
        ):

    if isinstance(position_generator_constructor, str):
        if position_generator_constructor=='default':
            position_generator_constructor = lambda: iter_pos_random(n_dim=2, rng=None)
        else:
            raise NotImplementedError(position_generator_constructor)

    is_checkpoint = Checkpoints(checkpoints)

    img = resize_image(smart_load_image(get_file('data/images/sistine_chapel.jpg', url='https://drive.google.com/uc?export=download&id=1g4HOxo2doBL6aPgYFoiqgLC8Mkinqao6')), width=2000, mode='preserve_aspect')

    img = img[img.shape[0]//2-image_cut_size[0]//2:img.shape[0]//2+image_cut_size[0]//2, img.shape[1]//2-image_cut_size[1]//2:img.shape[1]//2+image_cut_size[1]//2]  # TODO: Revert... this is just to test on a smaller version
    dbplot(img, 'full_img')

    model = model_constructor(batch_size, crop_size)

    duck = Duck()
    batched_bbox_generator = batchify_generator(list(
        iter_bboxes_from_positions(
            img_size=img.shape[:2],
            crop_size=crop_size,
            position_generator=position_generator_constructor(),
        ) for _ in range(batch_size)))

    t_start = time.time()
    for i, bboxes in enumerate(batched_bbox_generator):
        if n_iter is not None and i>=n_iter:
            break

        image_crops = (batch_crop(img=img, bboxes=bboxes).astype(np.float32))
        positions = np.array(bboxes)[:, [1, 0]] / (img.shape[0] - crop_size[0], img.shape[1] - crop_size[1]) - 0.5

        predicted_imgs, training_loss = model.train(positions, image_crops)

        pixel_error = np.abs(predicted_imgs - image_crops).mean()/255.

        duck[next, :] = dict(iter=i, pixel_error=pixel_error, elapsed=time.time()-t_start, training_loss=training_loss)

        if do_every('10s'):
            report = f'Iter: {i}, Pixel Error: {pixel_error:3g}, Mean Rate: {i/(time.time()-t_start):.3g}iter/s'
            print(report)
            with hold_dbplots():
                dbplot(image_crops, 'crops')
                dbplot(predicted_imgs, 'predicted_crops', cornertext=report)
        if is_checkpoint():
            yield duck


X = demo_train_just_vae_on_images_gqn.add_config_root_variant('deconv1', model_constructor = DeconvCropPredictor.get_constructor)
X32 = X.add_variant(filters = 32, n_iter=10000)
X64 = X.add_variant(filters = 64, n_iter=10000)
X128 = X.add_variant(filters = 128, n_iter=10000)
X256 = X.add_variant(filters = 256, n_iter=10000)

demo_train_just_vae_on_images_gqn.add_config_variant('gqn1', model_constructor = GQNCropPredictor.get_constructor)


if __name__ == '__main__':
    # X64.run()
    # X32.run()
    # X64.run()
    # X128.run()
    # demo_train_just_vae_on_images.browse()
    # demo_train_just_vae_on_images_gqn()
    demo_train_just_vae_on_images_gqn.browse(raise_display_errors=True)

    # demo_train_just_vae_on_images_gqn.get_variant('deconv1').run()
    # demo_train_just_vae_on_images_gqn.get_variant('gqn1').call()
