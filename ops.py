import tensorflow as tf
from tensorflow.python.framework import ops

# Implementation of conv layer with regularization
# Credit - https://github.com/bamos/dcgan-completion.tensorflow/blob/master/ops.py
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        reg = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        return conv, reg

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(x, W_shape, b_shape, name, padding='SAME'):
    W = weight_variable(W_shape)
    b = bias_variable([b_shape])
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)

def pool_layer(x):
    '''
    see description of build method
    '''
    with tf.device('/gpu:0'):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
    W = weight_variable(W_shape)
    b = bias_variable([b_shape])
    x_shape = tf.shape(x)
    out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

    return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b
