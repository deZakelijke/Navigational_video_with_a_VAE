from __future__ import print_function

from abc import abstractmethod

import numpy as np
import tensorflow as tf
import time
from argparse import Namespace
from tensorflow.python.training.adam import AdamOptimizer
from torch.distributions import LogNormal
from torch.nn.functional import binary_cross_entropy
from typing import Callable, Tuple, Iterator

from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record_view import get_timeseries_record_comparison_function, \
    timeseries_oneliner_function, get_timeseries_oneliner_function
from artemis.fileman.file_getter import get_file
from artemis.fileman.smart_io import smart_load_image
from artemis.general.checkpoint_counter import Checkpoints, do_every
from artemis.general.deferred_defaults import default
from artemis.general.duck import Duck
from artemis.general.image_ops import resize_image
from artemis.ml.tools.iteration import batchify_generator
from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes
from src.VAE_with_Disc import VAE, get_encoding_convnet
from src.gqn.gqn_draw import generator_rnn
from src.gqn.gqn_params import set_gqn_param, get_gqn_param
from src.peters_stuff.gqn_pose_predictor import query_pos_inference_rnn
from src.peters_stuff.image_crop_generator import iter_bboxes_from_positions, iter_pos_random, batch_crop


class IPositionPredictor(object):

    @abstractmethod
    def train(self, positions, image_crops):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, im):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_constructor(**kwargs) -> Callable[[int, Tuple[int,int]], 'IPositionPredictor']:
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


def _get_named_opt(name, parameters, learning_rate, weight_decay = 0.):
    import torch
    if name=='adam':
        return torch.optim.Adam(parameters, lr = learning_rate, betas = (0.5, 0.999), weight_decay=weight_decay)
    elif name=='rmsprop':
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name=='sgd':
        return torch.optim.SGD(parameters, lr = learning_rate, weight_decay=weight_decay)
    elif name=='sgd-mom':
        return torch.optim.SGD(parameters, lr = learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise NotImplementedError(name)


class ConvnetPositionPredictor(IPositionPredictor):

    def __init__(self, image_size, learning_rate=1e-3, weigth_decay = 0., filters=32, img_channels=3, use_batchnorm=True, opt = 'sgd'):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VAE(latent_dims=2, image_size=image_size, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm).to(self.device)
        self.opt = _get_named_opt(opt, parameters=self.model.parameters(), learning_rate=learning_rate, weigth_decay=weigth_decay)

    def train(self, image_crops, positions):
        import torch
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        positions = torch.Tensor(positions).to(self.device)
        mu, logvar = self.model.encode(var_image_crops)
        # loss = (.5 * logvar + (positions - mu)**2 / (2*torch.exp(logvar))).sum(dim=1).mean(dim=0)
        # loss = (.5 * logvar + (positions - mu)**2 / (2*torch.nn.functional.softplus(logvar))).sum(dim=1).mean(dim=0)
        loss = ((positions - mu)**2).sum(dim=1).mean(dim=0)
        loss.backward()
        self.opt.step()
        return mu.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def predict(self, image_crops):
        import torch
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        mu, logvar = self.model.encode(var_image_crops)
        return mu.detach().cpu().numpy()

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32, img_channels = 3, use_batchnorm = True, opt='sgd', weigth_decay=0.):
        return lambda batch_size, image_size: ConvnetPositionPredictor(image_size=image_size, learning_rate=learning_rate, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm, weigth_decay=weigth_decay)



class ConvnetPositionPredictor2(IPositionPredictor):

    def __init__(self, image_size, grid_size=(5, 5), learning_rate=1e-3, filters=32, img_channels=3, opt = 'sgd'):

        from torch import nn
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)

        class CoordConv2d(nn.Module):

            def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
                nn.Module.__init__(self)
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.coordconv = nn.Conv2d(2, out_channels, kernel_size, stride=stride, padding=padding)
                self.coords = None

            def forward(self, x):
                if self.coords is None:
                    sy, sx = x.size()[-2:]
                    self.coords = torch.stack(torch.meshgrid([torch.linspace(0, 1, sy), torch.linspace(0, 1, sx)]))[None].to('cuda' if torch.cuda.is_available() else 'cpu')
                return self.conv(x) + self.coordconv(self.coords)

        class PositionEstimationLayer(nn.Module):

            def __init__(self, grid_size, ranges):
                nn.Module.__init__(self)
                minrange, maxrange = ranges
                self.position_grid = torch.Tensor(np.array(np.meshgrid(*(np.linspace(minrange, maxrange, s) for s in grid_size))).reshape(2, -1)).to(device)  # (2, grid_size[0]*grid_size[1])

            def forward(self, x):
                weights = torch.nn.functional.softmax(x, dim=1)  # (n_samples, grid_size[0]*grid_size[1])
                est_pos = (weights[:, None, :]*self.position_grid[None, :, :]).sum(dim=2)
                return est_pos

        # Decoding layers
        n_final_channels = 2
        self.device = device
        self.model = nn.Sequential(
            CoordConv2d(img_channels, filters, 3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv2d(filters, filters, 3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv2d(filters, filters, 3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv2d(filters, filters, 3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv2d(filters, n_final_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(n_final_channels*64*64, grid_size[0]*grid_size[1]),
            PositionEstimationLayer(grid_size=grid_size, ranges=(-.5, .5))
        ).to(self.device)



        # self.fc  = nn.Linear(2*64*64, 2)
        self.opt = _get_named_opt(opt, parameters=self.model.parameters(), learning_rate=learning_rate)

    def train(self, image_crops, positions):
        import torch
        from torch.nn.functional import relu
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        positions = torch.Tensor(positions).to(self.device)

        est_pos = self.model(var_image_crops)


        loss = ((positions - est_pos)**2).sum(dim=1).mean(dim=0)
        loss.backward()
        self.opt.step()

        # mu, logvar = self.model.encode(var_image_crops)
        # loss = (.5 * logvar + (positions - mu)**2 / (2*torch.exp(logvar))).sum(dim=1).mean(dim=0)
        # # loss = (.5 * logvar + (positions - mu)**2 / (2*torch.nn.functional.softplus(logvar))).sum(dim=1).mean(dim=0)
        # loss = ((positions - mu)**2).sum(dim=1).mean(dim=0)
        # loss.backward()
        # self.opt.step()
        return est_pos.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def predict(self, image_crops):
        import torch
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        mu, logvar = self.model.encode(var_image_crops)
        return mu.detach().cpu().numpy()

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32, img_channels = 3, use_batchnorm = True, opt='sgd'):
        return lambda batch_size, image_size: ConvnetPositionPredictor2(image_size=image_size, learning_rate=learning_rate, filters=filters, img_channels=img_channels)



class ConvnetGridPredictor(IPositionPredictor):

    def __init__(self, image_size, learning_rate=1e-3, filters=32, img_channels=3, use_batchnorm=True, gridsize= (10, 10), opt='sgd'):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VAE(latent_dims=gridsize[0]*gridsize[1], image_size=image_size, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm).to(self.device)
        # self.opt = torch.optim.Adam(list(self.model.parameters()), lr = learning_rate, betas = (0.5, 0.999))
        self.opt = _get_named_opt(opt, parameters=self.model.parameters(), learning_rate=learning_rate)
        self.position_grid = torch.Tensor(np.array(np.meshgrid(*(np.linspace(-.5, .5, s) for s in gridsize))).reshape(2, -1)).to(self.device)  # (2, grid_size[0]*grid_size[1])

    def train(self, image_crops, positions):
        import torch
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        positions = torch.Tensor(positions).to(self.device)
        mu, logvar = self.model.encode(var_image_crops)
        weights = torch.nn.functional.softmax(mu, dim=1)  # (n_samples, grid_size[0]*grid_size[1])
        est_pos = (weights[:, None, :]*self.position_grid[None, :, :]).sum(dim=2)
        loss = ((positions - est_pos)**2).sum(dim=1).mean(dim=0)
        loss.backward()
        self.opt.step()
        return est_pos.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def predict(self, image_crops):
        raise NotImplementedError()

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32, img_channels = 3, use_batchnorm = True, opt='sgd', gridsize=(10, 10)):
        return lambda batch_size, image_size: ConvnetGridPredictor(image_size=image_size, learning_rate=learning_rate, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm, opt=opt, gridsize=gridsize)


class ConvnetGridPredictor2(IPositionPredictor):

    def __init__(self, image_size, learning_rate=1e-3, filters=32, weight_decay=0., img_channels=3, use_batchnorm=True, gridsize= (10, 10), opt='sgd'):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = get_encoding_convnet(image_channels=img_channels, filters=filters, use_batchnorm=use_batchnorm, grid_size=gridsize).to(self.device)
        # self.model = VAE(latent_dims=gridsize[0]*gridsize[1], image_size=image_size, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm).to(self.device)
        # self.opt = torch.optim.Adam(list(self.model.parameters()), lr = learning_rate, betas = (0.5, 0.999))
        self.opt = _get_named_opt(opt, parameters=self.model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
        self.position_grid = torch.Tensor(np.array(np.meshgrid(*(np.linspace(-.5, .5, s) for s in gridsize))).reshape(2, -1)).to(self.device)  # (2, grid_size[0]*grid_size[1])

    def train(self, image_crops, positions):
        import torch
        var_image_crops = torch.Tensor(imbatch_to_feat(image_crops, channel_first=True, datarange=(0, 1))).to(self.device)
        positions = torch.Tensor(positions).to(self.device)
        est_pos = self.model(var_image_crops)
        # weights = torch.nn.functional.softmax(mu, dim=1)  # (n_samples, grid_size[0]*grid_size[1])
        # est_pos = (weights[:, None, :]*self.position_grid[None, :, :]).sum(dim=2)
        loss = ((positions - est_pos)**2).sum(dim=1).mean(dim=0)
        loss.backward()
        self.opt.step()
        return est_pos.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def predict(self, image_crops):
        raise NotImplementedError()

    @staticmethod
    def get_constructor(learning_rate=1e-3, filters=32, img_channels = 3, use_batchnorm = True, opt='sgd', gridsize=(10, 10), weight_decay=0.):
        return lambda batch_size, image_size: ConvnetGridPredictor2(image_size=image_size, learning_rate=learning_rate, filters=filters, img_channels=img_channels, use_batchnorm=use_batchnorm, opt=opt, gridsize=gridsize, weight_decay=weight_decay)




def bbox_to_position(bboxes, image_size, crop_size):
    return np.array(bboxes)[:, [1, 0]] / (image_size[0] - crop_size[0], image_size[1] - crop_size[1]) - 0.5


def position_to_bbox(positions, image_size, crop_size, clip=False):

    if clip:
        positions = np.clip(positions, -.5, .5)

    unnorm_positions = ((np.array(positions)+.5)[:, [1, 0]] * (image_size[0] - crop_size[0], image_size[1] - crop_size[1])).astype(np.int)
    bboxes = np.concatenate([unnorm_positions, unnorm_positions+crop_size], axis=1)
    return bboxes



class GQNPositionPredictor(IPositionPredictor):
    """
    Seems to work... But it's a big slow beast.
    """

    def __init__(self, batch_size, image_size, enc_h=16, enc_w=16, rnn_params = {}):
        set_gqn_param('POSE_CHANNELS', 2)
        # enc_h, enc_w = get_gqn_param('ENC_HEIGHT'), get_gqn_param('ENC_WIDTH')
        g = Namespace()
        # g.positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
        g.target_frames = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
        g.target_positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
        g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, enc_h, enc_w, 1))
        g.mu_targ, _ = query_pos_inference_rnn(representations=g.representations, target_frames=g.target_frames, sequence_size=12, **rnn_params)
        g.loss = tf.reduce_mean((g.mu_targ-g.target_positions)**2)
        g.update_op = AdamOptimizer().minimize(g.loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.g = g
        self.sess = sess

    def train(self, image_crops, positions):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_positions, _, loss = self.sess.run([self.g.mu_targ, self.g.update_op, self.g.loss] , feed_dict={self.g.target_positions: positions, self.g.target_frames: image_crops})

        # with hold_dbplots(draw_every=10):  # Just check that data normalization is right
        #     dbplot(predicted_imgs, 'preed')
        #     dbplot(image_crops, 'crooops')
        return predicted_positions, loss

    def predict(self, positions):
        raise NotImplementedError()
        # predicted_imgs, _, loss = self.sess.run([self.g.mu_targ] , feed_dict={self.g.positions: positions})
        # return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss

    @staticmethod
    def get_constructor(enc_h=16, enc_w=16, rnn_params={}):
        return lambda batch_size, image_size: GQNPositionPredictor(batch_size=batch_size, image_size=image_size, rnn_params=rnn_params, enc_h=enc_h, enc_w=enc_w)



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

    if append_position_maps:
        position_maps = np.broadcast_to(feat_to_imbatch(np.array(np.meshgrid(*(np.linspace(0, 1, cs) for cs in crop_size)))[None], channel_first=True, datarange=(0, 1)), (batch_size, *crop_size, 2))

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

        if do_every('5s'):
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
# Xgqn_params = Xgqn.add_variant(rnn_params = {})
# Xgqn_params = Xgqn.add_variant(enc_h=4, enc_w = 4, rnn_params = dict(lstm_canvas_channels=64))
# Xgqn_params = Xgqn.add_variant(enc_h=4, enc_w = 4, )
Xgqn_params = Xgqn.add_variant(rnn_params = dict(lstm_canvas_channels=0, lstm_output_channels=64, generator_input_channels=0, inference_input_channels=0))

if __name__ == '__main__':
    Xgqn_params.run()
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
