from collections.__init__ import namedtuple

import tensorflow as tf
from argparse import Namespace
from tensorflow.python.training.adam import AdamOptimizer

from src.gqn.gqn_draw import generator_rnn
from src.gqn.gqn_params import set_gqn_param, get_gqn_param
from src.peters_stuff.crop_predictors import ICropPredictor, imbatch_to_feat, feat_to_imbatch
from src.peters_stuff.gqn_pose_predictor import convlstm_position_to_image_decoder
from src.peters_stuff.tf_helpers import TFGraphClass


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


class GQNCropPredictor2(ICropPredictor):

    def __init__(self, batch_size, image_size, n_maps=256, canvas_channels=256, sequence_size=12):
        # set_gqn_param('POSE_CHANNELS', 2)
        # enc_h, enc_w = get_gqn_param('ENC_HEIGHT'), get_gqn_param('ENC_WIDTH')
        g = Namespace()
        g.positions = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
        g.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, *image_size, 3))
        # g.representations = tf.zeros(dtype=tf.float32, shape=(batch_size, enc_h, enc_w, 1))
        g.mu_targ, g.var = convlstm_position_to_image_decoder(query_poses=g.positions, image_shape=image_size[:2] + (3,), cell_downsample=4, n_maps=n_maps, canvas_channels=canvas_channels, sequence_size=sequence_size)
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
    def get_constructor(n_maps=256, canvas_channels=256, sequence_size=12):
        return lambda batch_size, image_size: GQNCropPredictor2(batch_size=batch_size, image_size=image_size, n_maps=n_maps, canvas_channels=canvas_channels, sequence_size=sequence_size)


CropPredictorNodes = namedtuple('CropPredictorNodes', ['positions', 'predicted_crops', 'target_crops', 'loss', 'update_op', 'batch_size'])


class GQNCropPredictor3(TFGraphClass[CropPredictorNodes], ICropPredictor):

    def train(self, positions, image_crops):
        image_crops = imbatch_to_feat(image_crops, channel_first=False, datarange=(-1, 1))
        predicted_imgs, _, loss = self.sess.run([self.nodes.predicted_crops, self.nodes.update_op, self.nodes.loss] , feed_dict={self.nodes.positions: positions, self.nodes.target_crops: image_crops, self.nodes.batch_size: len(image_crops)})
        return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1)), loss

    def predict(self, positions):
        predicted_imgs, = self.sess.run([self.nodes.predicted_crops] , feed_dict={self.nodes.positions: positions, self.nodes.batch_size: len(positions)})
        return feat_to_imbatch(predicted_imgs, channel_first=False, datarange=(-1, 1))

    @staticmethod
    def get_constructor(n_maps=64, canvas_channels=64, sequence_size=12):
        def graph_constructor(batch_size, image_size):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Yes it's silly but we have to
            positions = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            targets = tf.placeholder(dtype=tf.float32, shape=(None, *image_size, 3))
            mu_targ, var = convlstm_position_to_image_decoder(query_poses=positions, batch_size=batch_size, image_shape=image_size[:2] + (3,), cell_downsample=4, n_maps=n_maps, canvas_channels=canvas_channels, sequence_size=sequence_size)
            loss = tf.reduce_mean((mu_targ-targets)**2)
            update_op = AdamOptimizer().minimize(loss)
            nodes = CropPredictorNodes(positions=positions, predicted_crops=mu_targ, target_crops=targets, loss=loss, update_op=update_op, batch_size=batch_size)
            return GQNCropPredictor3(nodes)
        return graph_constructor