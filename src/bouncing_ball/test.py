import tensorflow as tf
import torch


def test_tensorflow(batch_size, channels, x_dim, y_dim):

    xx_ones = tf.ones([batch_size, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    print(xx_ones.shape)
    xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    print(xx_range.shape)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([batch_size, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, -1)
    print(yy_ones.shape)
    yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size, 1])
    yy_range = tf.expand_dims(yy_range, 1)
    print(yy_range.shape)
    yy_channel = tf.matmul(yy_ones, yy_range)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel = tf.cast(xx_channel, 'float32') / (x_dim - 1)
    yy_channel = tf.cast(yy_channel, 'float32') / (y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    print(xx_channel.shape)
    print(yy_channel.shape)
    ret = tf.concat([xx_channel, yy_channel], axis=-1)

    print(ret.shape)


def test_torch(batch_size, channels, x_dim, y_dim):

        xx_ones = torch.ones([batch_size, x_dim], dtype=torch.long)
        xx_ones = xx_ones.unsqueeze(-1)
        print(xx_ones.shape)
        xx_range = torch.arange(x_dim).unsqueeze(0).repeat([batch_size, 1])
        xx_range = xx_range.unsqueeze(1)
        print(xx_range.shape)
        xx_channel = torch.matmul(xx_ones, xx_range) 
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([batch_size, y_dim], dtype=torch.long)
        yy_ones = yy_ones.unsqueeze(-1)
        print(yy_ones.shape)
        yy_range = torch.arange(y_dim).unsqueeze(0).repeat([batch_size, 1])
        yy_range = yy_range.unsqueeze(1)
        print(yy_range.shape)
        yy_channel = torch.matmul(yy_ones, yy_range) 
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel.float()
        yy_channel.float()
        xx_channel = xx_channel / (x_dim - 1) * 2 - 1
        yy_channel = yy_channel / (y_dim - 1) * 2 - 1

        print(xx_channel.shape)
        print(yy_channel.shape)
        ret = torch.cat([xx_channel, yy_channel], dim=-1)
        print(ret.shape)


if __name__ == "__main__":
    batch = 64
    x = 30
    y = 30
    c = 3
    print("Tensorflow")
    test_tensorflow(batch, c, x, y)
    print("Pytorch")
    test_torch(batch, c, x, y)
