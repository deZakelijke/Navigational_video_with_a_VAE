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
            y = tf.layers.conv2d(
                x,
                filters=4,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=True,
                activation=None,
                name='y'
              )
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


MyNodes = namedtuple('MyNodes', ['x', 'y'])


def test_graph_save_with_namedtuple():

    im = np.random.randn(5, 10, 10, 3)
    model_path = get_artemis_data_path('tests/test_save_dir/model', make_local_dir=True)

    rmtree(os.path.split(model_path)[0])

    with tf.Graph().as_default():

        def define_nodes():
            x = tf.placeholder(shape=im.shape, dtype=tf.float32, name='x')
            y = tf.layers.conv2d(
                x,
                filters=4,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=True,
                activation=None,
                name='y'
              )
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
# all_ops_in_graph


def test_duplicate_graph():

    im = np.random.randn(5, 10, 10, 3)
    obj = MyConvObj.from_graph(im_shape= im.shape)

    x, y = obj.nodes.x, obj.nodes.y


    sess = tf.Session()

    sess.runsess.run(tf.global_variables_initializer())

    sess.run(y, feed_dict={x: im})

    # ser_dir = obj.dump()

    # obj2 = TFGraphClass.load(ser_dir)  # type: MyConvObj

    x1, y1 = replicate_subgraph(inputs=obj.nodes.x, outputs = obj.nodes.y)
    # raise NotImplementedError('Gave up')


    sess.run(y1, feed_dict={x1: im})





# def test_keep_it_simple():
#     im = np.random.randn(5, 10, 10, 3)
#
#     x1 = tf.placeholder(shape=im.shape, dtype=tf.float32, name='x1')
#     y1 = tf.layers.conv2d(x1, filters=4)
#
#
#     x2 = tf.placeholder(shape=im.shape, dtype=tf.float32, name='x2')
#     y2 = tf.layers.conv2d(x1, filters=4)

#     y2 = tf.layers.conv2d(x1, filters=4)



if __name__ == '__main__':
    # test_graph_save()
    # test_graph_save_with_namedtuple()
    # test_serializable_obj()
    test_duplicate_graph()
