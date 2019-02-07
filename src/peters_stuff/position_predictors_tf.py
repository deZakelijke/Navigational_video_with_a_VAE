from collections.__init__ import namedtuple

import tensorflow as tf
from argparse import Namespace
from tensorflow.python.training.adam import AdamOptimizer

from src.gqn.gqn_params import set_gqn_param
from src.peters_stuff.gqn_pose_predictor import query_pos_inference_rnn, convlstm_image_to_position_encoder
from src.peters_stuff.position_predictors import IPositionPredictor, imbatch_to_feat
from src.peters_stuff.tf_helpers import TFGraphClass


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