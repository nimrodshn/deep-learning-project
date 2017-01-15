import copy
import os
import datetime
import time
import argparse
import tensorflow as tf
import numpy as np
from data.DataHandeling import DataSets
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

"""
FLAGS - an easy way to share constants variables between functions
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
# flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
# flags.DEFINE_float('regularization_weight',5e-4, 'L2 Norm regularization weight.')
# flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
# flags.DEFINE_integer('print_test', 1000, 'Print test frequency')
# flags.DEFINE_integer('print_train', 100, 'Print train frequency')

# Please do not change those two flags
flags.DEFINE_string('train_dir',
						   './train_results/',
						   """Directory where to write event logs """
						   """and checkpoints.""")
flags.DEFINE_string('data_dir',
						   './data/',
						   """Directory of input data for the network """)

file_names = ['train', 'test', 'val']

DIMS_IN = (64, 64, 1)
DIMS_OUT = (64, 64, 1)
TEST_AMOUNT = 478

# File for stdout
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % datetime.datetime.now()), 'w')


data_sets = DataSets(filenames=file_names, base_folder=FLAGS.data_dir, image_size=DIMS_IN)
data_set_train = data_sets.data['train']
data_set_val = data_sets.data['val']
data_set_test = data_sets.data['test']
model = Sequential()
model.add(Convolution2D(32, 3, 3 ,input_shape=(1, 64, 64), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Convolution2D(1, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
trainSet, labelSet = data_set_train.get_image()
with tf.Session().as_default():
	trainSetNumPy = trainSet.eval()
	labelSetNumPy = labelSet.eval()

# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(trainSetNumPy, labelSetNumPy, batch_size=32, nb_epoch=epochs, verbose=1)
# print(model.summary())
