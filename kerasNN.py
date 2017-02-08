import copy
import os
import datetime
import time
import argparse
import tensorflow as tf
import numpy as np
from data.DataHandeling import DataSets
from keras.models import Sequential, Model
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

from keras.layers.convolutional import Convolution2D, Deconvolution2D, AtrousConvolution2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend as K
from Val_Callback import Val_Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from DiceMetric import dice_coeff, dice_coeff_loss
from binary_cross import binary_cross_entropy
from Schedulers import Schedulers
from MyDataManipulator import plot_pairwise_train, plot_pairwise_val, read_train_val_data


def Net(w_regularize):
	# Input Image 64x64
	main_input = Input(shape=(64, 64,1))
	x = Convolution2D(8, 3, 3 , init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(w_regularize))(main_input)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Input Image 32x32
	x = Convolution2D(16, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	
	# Input Image 16x16
	x = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Activation('relu')(x)
	# Input Image 8x8
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	# Inputs Image 16x16
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(16, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	# Inputs Image 32x32
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(1, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('sigmoid')(x)
	model = Model(input=[main_input], output=[x])
	print(model.summary())
	return model

def trainNet(model,epochs, lrate, knee_epoch, batch_size_train, train_set_numpy,train_label_set_numpy,val_set_numpy,val_label_set_numpy):
	# Compile model

	adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# sgd = SGD(lr=lrate, momentum=0.9, decay=0.0, nesterov=False)

	# DEBUG
	print train_set_numpy.shape
	print train_label_set_numpy.shape

	validation_callback = Val_Callback(val_data=(val_set_numpy,val_label_set_numpy),train_data=(train_set_numpy, train_label_set_numpy) , model=model)

	scheduler = Schedulers(lrate, knee_epoch)
	lr_scheduler_func = lambda epochNumber : scheduler.lr_scheduler(epochNumber)
	reduce_lr = LearningRateScheduler(lr_scheduler_func)
	#early_stop = EarlyStopping(monitor='val_dice_coeff', min_delta=0, patience=10, verbose=0, mode='auto')

	#model.compile(loss=[felix_loss], optimizer=adam, metrics=[dice_coeff]) #metrics=['accuracy'])
	model.compile(loss=[dice_coeff_loss], optimizer=adam, metrics=[dice_coeff]) #metrics=['accuracy'])
	model.fit(train_set_numpy, train_label_set_numpy, callbacks=[validation_callback, reduce_lr] , validation_data = (val_set_numpy,val_label_set_numpy), batch_size=batch_size_train, nb_epoch=epochs, verbose=1)
	print(model.summary())


def getDataSet():

	train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy = read_train_val_data()

	return train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy

def runNet():
	w_regularize = 5e-4
	epochs = 200
	learning_rate = 0.01
	batch_size_train = 20
	knee_epoch = 4;
	train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy = getDataSet()
	model = Net(w_regularize)
	trainNet(model, epochs, learning_rate, knee_epoch, batch_size_train, train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy)
