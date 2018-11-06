from abc import abstractmethod
from argparse import Namespace
from collections import namedtuple
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from src.VAE_with_Disc import VAE
from src.gqn.gqn_params import set_gqn_param
from src.peters_stuff.gqn_pose_predictor import query_pos_inference_rnn, convlstm_image_to_position_encoder
from src.peters_stuff.tf_helpers import TFGraphClass


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


class GQNPositionPredictor2(IPositionPredictor):
    """
    Seems to work... But it's a big slow beast.
    """

    def __init__(self, batch_size, image_size, cell_downsample=4, n_maps=32):
        set_gqn_param('POSE_CHANNELS', 2)
        g = Namespace()
        g.target_frames = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
        g.target_positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
        g.mu_targ, _ = convlstm_image_to_position_encoder(image=g.target_frames, cell_downsample=cell_downsample, n_maps=n_maps)
        g.loss = tf.reduce_mean((g.mu_targ-g.target_positions)**2)
        g.update_op = AdamOptimizer().minimize(g.loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.g = g
        self.sess = sess

    def train(self, image_crops, positions):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_positions, _, loss = self.sess.run([self.g.mu_targ, self.g.update_op, self.g.loss] , feed_dict={self.g.target_positions: positions, self.g.target_frames: image_crops})
        return predicted_positions, loss

    def predict(self, positions):
        raise NotImplementedError()

    @staticmethod
    def get_constructor(n_maps=32):
        return lambda batch_size, image_size: GQNPositionPredictor2(batch_size=batch_size, image_size=image_size, n_maps=n_maps)


PositionPredictorNodes = namedtuple('PositionPredictorNodes', ['batch_size', 'crops', 'predicted_positions', 'target_positions', 'loss', 'update_op'])


class GQNPositionPredictor3(TFGraphClass[PositionPredictorNodes]):
    """

    """

    def train(self, image_crops, positions):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_positions, _, loss = self.sess.run([self.nodes.predicted_positions, self.nodes.update_op, self.nodes.loss] , feed_dict={self.nodes.target_positions: positions, self.nodes.crops: image_crops, self.nodes.batch_size: len(image_crops)})
        return predicted_positions, loss

    def predict(self, image_crops):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_positions, = self.sess.run([self.nodes.predicted_positions], feed_dict={self.nodes.crops: image_crops, self.nodes.batch_size: len(image_crops)})
        return predicted_positions

    @staticmethod
    def get_constructor(n_maps=32, cell_downsample=4):

        def constructor(batch_size, image_size):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Yes it's silly but we have to
            crops = tf.placeholder(dtype=tf.float32, shape=(None, *image_size, 3))
            target_positions = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            predicted_positions, _ = convlstm_image_to_position_encoder(image=crops, batch_size=batch_size, cell_downsample=cell_downsample, n_maps=n_maps, n_pose_channels=2)
            loss = tf.reduce_mean((predicted_positions-target_positions)**2)
            update_op = AdamOptimizer().minimize(loss)
            return GQNPositionPredictor3(nodes=PositionPredictorNodes(crops=crops, predicted_positions=predicted_positions, target_positions=target_positions, loss=loss, update_op=update_op, batch_size=batch_size))

        return constructor
