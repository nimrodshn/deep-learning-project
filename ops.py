import tensorflow as tf
from tensorflow.python.framework import ops

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv_layer(x, W_shape, b_shape, name, padding='SAME'):
	W = weight_variable(W_shape)
	b = bias_variable([b_shape])
	conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)
	reg = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
	return conv,reg

def pool_layer(x):
	'''
	see description of build method
	'''
	with tf.device('/gpu:0'):
		return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def unravel_argmax(argmax, shape):
	output_list = []
	output_list.append(argmax // (shape[2] * shape[3]))
	output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
	return tf.pack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
	argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
	output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

	height = tf.shape(output)[0]
	width = tf.shape(output)[1]
	channels = tf.shape(output)[2]
	t1 = tf.to_int64(tf.range(channels))
	t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
	t1 = tf.reshape(t1, [-1, channels])
	t1 = tf.transpose(t1, perm=[1, 0])
	t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

	t2 = tf.squeeze(argmax)
	t2 = tf.pack((t2[0], t2[1]), axis=0)
	t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

	t = tf.concat(3, [t2, t1])
	indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

	x1 = tf.squeeze(x)
	x1 = tf.reshape(x1, [-1, channels])
	x1 = tf.transpose(x1, perm=[1, 0])
	values = tf.reshape(x1, [-1])

	delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
	return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)


def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
	W = weight_variable(W_shape)
	b = bias_variable([b_shape])
	x_shape = tf.shape(x)
	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
	deconv = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b
	reg = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
	return deconv,reg
	
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		weights = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME')
		biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.nn.bias_add(conv, biases)
		reg = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
		return conv, reg