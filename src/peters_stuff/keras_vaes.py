from tensorflow.python.keras.losses import binary_crossentropy

from src.peters_stuff.keras_graph_model import GraphModel
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer, LambdaLayer

class VAENodeNames:
    X_SAMPLE = 'x_sample'
    Z_MU = 'z_mu'
    Z_LOGSIG = 'z_logvar'
    Z_SAMPLE = 'z_sample'
    X_MU = 'x_mu'
    X_LOGSIG = 'x_var'
    ELBO = 'elbo'
    UPDATE_OP = 'update_op'
    # TRAIN_OP = 'train_op'
    # BATCH_SIZE = 'batch_size'

V = VAENodeNames


class ReparametrizationLayer(Layer):

    def __init__(self,):
        self.shape = None
        super(ReparametrizationLayer, Layer).__init__(self)

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, inputs):
        mu, logvar = inputs
        sample = tf.random_normal(shape = self.shape)*tf.sqrt(tf.exp(logvar)) + mu
        return sample


def vae_loss(x, z_mu, z_log_sigma, x_mu, x_log_sigma = None, is_bernoulli = True):
    if is_bernoulli:
        recon_loss = -tf.distributions.Bernoulli(logits=x_mu).log_prob(x)
    else:
        recon_loss = -tf.distributions.Normal(loc=x_mu, scale = tf.exp(x_log_sigma)).log_prob(x)
    kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_sigma - z_mu**2 - tf.exp(z_log_sigma), axis=-1)
    return recon_loss + kl_loss


class VAELossLayer(Layer):

    def call(self, args):
        x, z_mu, z_log_sigma, x_mu, x_log_sigma = args
        return vae_loss(x=x, z_mu=z_mu, z_log_sigma=z_log_sigma, x_mu=x_mu, x_log_sigma=x_log_sigma)



def make_keras_vae_model(encoder, decoder, is_bernoulli):

    return GraphModel(graph = {
        (V.X_SAMPLE, (V.Z_MU, V.Z_LOGSIG)): encoder,
        ((V.Z_MU, V.Z_LOGSIG), V.Z_SAMPLE): ReparametrizationLayer(),
        (V.Z_SAMPLE, V.X_MU if is_bernoulli else (V.X_MU, V.X_LOGSIG)): decoder,
        ((V.X_SAMPLE, V.Z_MU, V.Z_LOGSIG, *((V.X_MU, ) if is_bernoulli else (V.X_MU, V.X_LOGSIG))), V.ELBO): LambdaLayer(),
        (V.ELBO, V.UPDATE_OP): ,
        },
        input_names=[V.X_SAMPLE],
        output_names=[V.UPDATE_OP],
        )


def get_keras_vae_model()