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
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend as K
from Val_Callback import Val_Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from DiceMetric import dice_coeff, dice_coeff_loss

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
model.add(Convolution2D(4, 3, 3 , init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(4, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Input Image 32x32
model.add(Convolution2D(8, 3, 3 , init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Input Image 16x16
model.add(Convolution2D(16, 3, 3 , init='he_normal', border_mode='same'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, init='he_normal', border_mode='same'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Input Image 8x8
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(16, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, init='he_normal', border_mode='same'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

#Input Image 16x16
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(8, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, init='he_normal', border_mode='same'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

#Input Image 32x32
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(4, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(5e-4)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(1, 3, 3, init='he_normal', border_mode='same'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('sigmoid'))



#model.add(Deconvolution2D(1, 3, 3, output_shape=(1,64,64,1), activation='softplus', border_mode='same', input_shape=(1,1,128)))
print(model.summary())

# Compile model

epochs = 25
lrate = 0.01
batch_size_train = 32;

adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
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

validation_callback = Val_Callback(val_data=(val_set_numpy,val_label_set_numpy),model=model)

reduce_lr = ReduceLROnPlateau(monitor='val_dice_coeff', factor=0.2,
                  patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_dice_coeff', min_delta=0, patience=2, verbose=0, mode='auto')

model.compile(loss=[dice_coeff_loss], optimizer=adam, metrics=[dice_coeff]) #metrics=['accuracy'])
model.fit(train_set_numpy, train_label_set_numpy, callbacks=[validation_callback, reduce_lr, early_stop] , validation_data = (val_set_numpy,val_label_set_numpy), batch_size=batch_size_train, nb_epoch=epochs, verbose=1)
print(model.summary())
