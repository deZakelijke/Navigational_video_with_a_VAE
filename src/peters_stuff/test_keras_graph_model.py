from shutil import rmtree

import numpy as np
import tensorflow as tf
import os

from artemis.fileman.local_dir import get_artemis_data_path
from src.peters_stuff.keras_graph_model import GraphModel


def test_graph_model():

    im1 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    im2 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)
    rmtree(os.path.split(model_path)[0])
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    model = GraphModel(
        graph = {('x', 'y'): tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='SAME')},
        inputs=[('x', tf.keras.Input(shape=im1.shape[1:], dtype=tf.float32))]
        )

    tf.keras.models.save_model(model, model_path)
    model = tf.keras.models.load_model(model_path)

    y1a = model.predict(im1)
    y2a = model.predict(im2)

    firstloss = ((y2a-y1a)**2).sum()

    model2 = GraphModel(
        graph = {
            ('x1', 'y1'): model,
            ('x2', 'y2'): model,
            (('y1', 'y2'), 'loss'): tf.keras.layers.Lambda(lambda yy: tf.reduce_sum((yy[0]-yy[1])**2))
        },
        inputs = [
            ('x1', tf.keras.Input(shape=im1.shape[1:], dtype=tf.float32)),
            ('x2', tf.keras.Input(shape=im2.shape[1:], dtype=tf.float32))
            ],
        output_names='loss'
    )

    print(model2.summary())
    lossval = model2.predict([im1, im2])
    assert np.allclose(firstloss, lossval)
    model3 = model2.get_subgraph(input_names='x1', output_names='y1')
    y3a = model3.predict(im1)
    assert np.allclose(y1a, y3a)


if __name__ == '__main__':
    test_graph_model()

