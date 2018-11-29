from functools import partial
from os import makedirs
from shutil import rmtree

import numpy as np
import tensorflow as tf
import os

from tensorflow.python.training.adam import AdamOptimizer

from artemis.fileman.local_dir import get_artemis_data_path
from src.peters_stuff.keras_graph_model import GraphModel


def test_graph_model():

    im1 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    im2 = np.random.randn(5, 10, 10, 3).astype(np.float32)

    model_dir = get_artemis_data_path('tests/test_save_dir/', make_local_dir=True)
    rmtree(os.path.split(model_dir)[0])
    makedirs(model_dir)

    model = GraphModel(
        graph = {('x', 'y'): [tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='SAME'), tf.keras.layers.ReLU()]},
        )

    y1a = model.predict(im1)
    y2a = model.predict(im2)

    path1 = os.path.join(model_dir, 'model1.keras')

    tf.keras.models.save_model(model, path1)
    model = tf.keras.models.load_model(path1)

    numpy_computed_loss = ((y2a-y1a)**2).sum()

    model2 = GraphModel(
        graph = {
            ('x1', 'y1'): model,
            ('x2', 'y2'): model,
            (('y1', 'y2'), 'loss'): tf.keras.layers.Lambda(lambda yy: tf.reduce_sum((yy[0]-yy[1])**2))
        },
        input_names = ['x1', 'x2'],
        output_names='loss'
    )
    lossval = model2.predict([im1, im2])

    path2 = os.path.join(model_dir, 'model1.keras')
    tf.keras.models.save_model(model2, path2)
    model2 = tf.keras.models.load_model(path2)

    assert np.allclose(numpy_computed_loss, lossval)
    model3 = model2.get_subgraph(input_names='x1', output_names='y1')
    y3a = model3.predict(im1)
    assert np.allclose(y1a, y3a)


def test_grain_keras_graph_model():

    x = np.random.randn(10, 5)
    y = (np.random.rand(10, 4)>0.5).astype(np.float)
    model = GraphModel(graph = {
        ('x', 'pred'): tf.keras.layers.Dense(units=4),
        (('y', 'pred'), 'loss'): tf.keras.losses.binary_crossentropy,
        # ('loss', 'update'): tf.keras.optimizers.Adam(),
        },
        input_names=['x', 'y'],
        output_names='loss',
        inputs_like = [x, y]
        )

    opt_model = GraphModel({
        (('x', 'y'), 'loss'): model,
        ('loss', 'train_op'): partial(AdamOptimizer().minimize, var_list = model.trainable_weights)
        },
        input_names=['x', 'y'],
        output_names='train_op',
        inputs_like = [x, y]
        )

    opt_model.predict([x, y])

    model.fit()

if __name__ == '__main__':
    # test_graph_model()
    test_grain_keras_graph_model()