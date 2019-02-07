import pickle
from collections import namedtuple

from shutil import rmtree
import numpy as np
import tensorflow as tf
from artemis.fileman.local_dir import get_artemis_data_path
import os

from src.peters_stuff.tf_helpers import save_model_and_graph, load_model_and_graph, TFGraphClass, hold_loading_scope, \
    replicate_subgraph


def test_graph_save():

    im = np.random.randn(5, 10, 10, 3)
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    rmtree(os.path.split(model_path)[0])

    with tf.Graph().as_default():

        def define_nodes():
            x = tf.placeholder(shape=im.shape, dtype=tf.float32, name='x')
            y = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=1, padding='SAME', use_bias=True, activation=None, name='y')
            return {'x': x, 'y': y}

        def first_run(x):
            nodes = define_nodes()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            y = sess.run(nodes['y'], feed_dict={nodes['x']: x})
            model_path = save_model_and_graph(sess=sess, nodes=nodes)
            return y, model_path

        y1, model_path = first_run(x=im)

        def second_run(model_path, x):
            nodes, sess = load_model_and_graph(model_path=model_path, scope='aaa')
            y = sess.run(nodes['y'], feed_dict={nodes['x']: x})
            return y

        y2 = second_run(model_path=model_path, x=im)

        assert np.allclose(y1, y2)


def test_graph_save_and_multi_load_with_raw_tf():

    im1 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    im2 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)
    rmtree(os.path.split(model_path)[0])
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    x = tf.placeholder(shape=im1.shape, dtype=tf.float32, name='x')
    y = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=1, padding='SAME', use_bias=True, activation=None, name='y')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model_path = save_model_and_graph(sess=sess, nodes={'x': x, 'y': y})
    y1a = sess.run(y, feed_dict={x: im1})
    y2a = sess.run(y, feed_dict={x: im2})
    firstloss = ((y2a-y1a)**2).sum()

    # Second: Create a new graph and insert 2 copies of the first graph in:
    x1 = tf.placeholder(shape=im1.shape, dtype=tf.float32, name='x1')
    x2 = tf.placeholder(shape=im1.shape, dtype=tf.float32, name='x2')
    newsess = tf.Session()
    nodes, sess = load_model_and_graph(model_path, sess=newsess)
    _, y1b = replicate_subgraph(inputs=[nodes['x']], new_inputs=[x1], outputs=nodes['y'])  # FAILS ON THIS LINE WITH SOME KIND OF FailedPreconditionError
    _, y2b = replicate_subgraph(inputs=[nodes['x']], new_inputs=[x1], outputs=nodes['y'])
    loss = tf.reduce_sum((y1b-y2b)**2)
    lossval = sess.run(loss, feed_dict={x1: im1, x2:im2})
    assert np.allclose(firstloss, lossval)


def test_graph_save_and_multi_load_with_keras():

    im1 = np.random.randn(5, 10, 10, 3).astype(np.float32)
    im2 = np.random.randn(5, 10, 10, 3).astype(np.float32)

    rmtree(get_artemis_data_path('tests/test_save_dir/model'))
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    with tf.Graph().as_default():

        x = tf.keras.layers.Input(shape=im1.shape[1:], dtype=tf.float32)
        layer = tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='SAME')
        y = layer(x)
        model = tf.keras.models.Model(inputs = [x], outputs=[y])
        tf.keras.models.save_model(model, model_path)

        model = tf.keras.models.load_model(model_path)

        y1a = model.predict(im1)
        y2a = model.predict(im2)

        firstloss = ((y2a-y1a)**2).sum()

        x1 = tf.keras.Input(shape=im1.shape[1:], dtype=tf.float32, name='x1')
        x2 = tf.keras.Input(shape=im2.shape[1:], dtype=tf.float32, name='x2')
        y1 = model(x1)
        y2 = model(x2)
        losslayer = tf.keras.layers.Lambda(lambda yy: tf.reduce_sum((yy[0]-yy[1])**2))
        loss = losslayer([y1, y2])

        model2 = tf.keras.models.Model(inputs=[x1, x2], outputs=[loss])

        print(model2.summary())
        lossval = model2.predict([im1, im2])
        assert np.allclose(firstloss, lossval)

        model3 = tf.keras.Model(inputs=x1, outputs=y1)

        y3a = model3.predict(im1)
        assert np.allclose(y1a, y3a)





# def test_graph_save_and_multi_load_with_tflearn():
#
#     im1 = np.random.randn(5, 10, 10, 3).astype(np.float32)
#     im2 = np.random.randn(5, 10, 10, 3).astype(np.float32)
#     model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)
#
#     rmtree(os.path.split(model_path)[0])
#     model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)
#
#     with tf.Graph().as_default():
#
#         x = tflearn.input_data(shape =im1.shape[1:], dtype=tf.float32)
#
#         y = tflearn.conv_2d(x, nb_filter=4, filter_size=4)
#
#         model = tflearn.models.DNN.load(model_path)
#
#         model.save(model_path)
#
#
#         model = tflearn.models.absolute_import(model_path)

        # x = tf.keras.layers.Input(shape=im1.shape[1:], dtype=tf.float32)
        # layer = tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='SAME')
        # y = layer(x)
        # model = tf.keras.models.Model(inputs = [x], outputs=[y])
        # tf.keras.models.save_model(model, model_path)
        #
        # model = tf.keras.models.load_model(model_path)
        #
        # y1a = model.predict(im1)
        # y2a = model.predict(im2)
        #
        # firstloss = ((y2a-y1a)**2).sum()
        #
        # x1 = tf.keras.Input(shape=im1.shape[1:], dtype=tf.float32, name='x1')
        # x2 = tf.keras.Input(shape=im2.shape[1:], dtype=tf.float32, name='x2')
        # y1 = model(x1)
        # y2 = model(x2)
        # losslayer = tf.keras.layers.Lambda(lambda yy: tf.reduce_sum((yy[0]-yy[1])**2))
        # loss = losslayer([y1, y2])
        #
        # model2 = tf.keras.models.Model(inputs=[x1, x2], outputs=[loss])
        # lossval = model2.predict([im1, im2])
        # assert np.allclose(firstloss, lossval)


MyNodes = namedtuple('MyNodes', ['x', 'y'])


def test_graph_save_with_namedtuple():

    im = np.random.randn(5, 10, 10, 3)
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    rmtree(os.path.split(model_path)[0])

    with tf.Graph().as_default():

        def define_nodes():
            x = tf.placeholder(shape=im.shape, dtype=tf.float32, name='x')
            y = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=1, padding='SAME', use_bias=True, activation=None, name='y')
            return MyNodes(x=x, y=y)

        def first_run(x):
            nodes = define_nodes()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            y = sess.run(nodes.y, feed_dict={nodes.x: x})
            model_path = save_model_and_graph(sess=sess, nodes=nodes)
            return y, model_path

        y1, model_path = first_run(x=im)

        def second_run(model_path, x):
            nodes, sess = load_model_and_graph(model_path=model_path)
            y = sess.run(nodes.y, feed_dict={nodes.x: x})
            return y

        y2 = second_run(model_path=model_path, x=im)

        assert np.allclose(y1, y2)


class MyConvObj(TFGraphClass[MyNodes]):

    def __call__(self, x):
        y = self.sess.run(self.nodes.y, feed_dict={self.nodes.x: x})
        return y

    @classmethod
    def from_graph(cls, im_shape):
        x = tf.placeholder(shape=im_shape, dtype=tf.float32, name='x')
        y = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=1, padding='SAME', use_bias=True, activation=None, name='y')
        nodes = MyNodes(x=x, y=y)
        return MyConvObj(nodes = nodes)


def test_serializable_obj():

    im = np.random.randn(5, 10, 10, 3)
    obj = MyConvObj.from_graph(im_shape= im.shape)

    y1 = obj(im)

    # Option 1: save/load manually with pickle
    ser = pickle.dumps(obj)
    with hold_loading_scope('bbb'):
        obj2 = pickle.loads(ser)
    ser = pickle.dumps(obj)
    y2 = obj2(im)
    assert np.allclose(y1, y2)

    # Option 2: More convenient interface with dump
    ser_dir = obj.dump()
    obj3 = TFGraphClass.load(ser_dir, scope='aaa')
    y3 = obj3(im)
    assert np.allclose(y1, y3)


def test_duplicate_graph():

    im = np.random.randn(5, 10, 10, 3)
    obj = MyConvObj.from_graph(im_shape= im.shape)

    x, y = obj.nodes.x, obj.nodes.y


    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    sess.run(y, feed_dict={x: im})

    # ser_dir = obj.dump()

    # obj2 = TFGraphClass.load(ser_dir)  # type: MyConvObj

    x1, y1 = replicate_subgraph(inputs=obj.nodes.x, outputs = obj.nodes.y)
    # raise NotImplementedError('Gave up')


    sess.run(y1, feed_dict={x1: im})


if __name__ == '__main__':
    # test_graph_save()
    # test_graph_save_with_namedtuple()
    # test_serializable_obj()
    # test_graph_save_and_multi_load_with_raw_tf()
    # test_graph_save_and_multi_load_with_tflearn()
    # test_graph_save_and_multi_load_with_keras()
    test_duplicate_graph()
