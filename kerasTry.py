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
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
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
# Input Image 64x64
model.add(Convolution2D(4, 3, 3 ,input_shape=(64, 64,1), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Convolution2D(4, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Input Image 32x32
model.add(Convolution2D(8, 3, 3 ,border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Input Image 16x16
model.add(Convolution2D(16, 3, 3 ,border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(UpSampling2D(size=(8,8)))
model.add(Convolution2D(1, 16, 16, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
#model.add(Deconvolution2D(1, 3, 3, output_shape=(1,64,64,1), activation='softplus', border_mode='same', input_shape=(1,1,128)))
print(model.summary())

# Compile model

epochs = 25
lrate = 0.01
decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
train_set, train_label_set = data_set_train.get_batch(batch_size=4815) # get all training data set hahahahaahs
val_set, val_label_set = data_set_val.get_batch(batch_size=467)
# Start tensorflow session
sess =  tf.Session()
tf.train.start_queue_runners(sess=sess)
init = tf.initialize_all_variables()
sess.run(init)
# Convert tensors to numpy arrays
train_set_numpy = sess.run(train_set)
train_label_set_numpy = sess.run(train_label_set)

val_set_numpy = sess.run(val_set)
val_label_set_numpy = sess.run(val_label_set)

# DEBUG
print train_set_numpy.shape
print train_label_set_numpy.shape

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_set_numpy, train_label_set_numpy, validation_data = (val_set_numpy,val_label_set_numpy), batch_size=10, nb_epoch=epochs, verbose=1)
print(model.summary())
