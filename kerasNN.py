import copy
import os
import datetime
import time
import argparse
import tensorflow as tf
import numpy as np
import NNmodels as NetTypes

from keras.optimizers import SGD, Adam
from keras import backend as K
from Val_Callback import Val_Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from DiceMetric import dice_coeff, dice_coeff_loss
from binary_cross import binary_cross_entropy
from MyDataManipulator import plot_pairwise_train, plot_pairwise_val, read_train_val_data


def Net(w_regularize, net_type):
	print net_type
	if net_type == 'FCN':
		model = NetTypes.modelFCN(w_regularize)
	if net_type == 'ResNetFCN':
		model = NetTypes.modelResNetFCN(w_regularize)
	if net_type == 'EncoderDecoder':
		model = NetTypes.modelEncoderDecoder(w_regularize)
	if net_type == 'EncoderDecoderResNet':
		model = NetTypes.modelEncoderDecoderResNet(w_regularize)
	return model

	

def trainNet(model,epochs, lrate, batch_size_train, train_set_numpy,train_label_set_numpy,val_set_numpy,val_label_set_numpy):
	# Compile model

	adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# sgd = SGD(lr=lrate, momentum=0.9, decay=0.0, nesterov=False)

	validation_callback = Val_Callback(val_data=(val_set_numpy,val_label_set_numpy),train_data=(train_set_numpy, train_label_set_numpy) , model=model)

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto')

	model.compile(loss=[dice_coeff_loss], optimizer=adam, metrics=[dice_coeff]) #metrics=['accuracy'])
	model.fit(train_set_numpy, train_label_set_numpy, callbacks=[validation_callback, reduce_lr] , validation_data = (val_set_numpy,val_label_set_numpy), batch_size=batch_size_train, nb_epoch=epochs, verbose=1)
	print(model.summary())


def getDataSet():

	train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy = read_train_val_data()

	return train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy

def runNet():
	w_regularize = 5e-4
	epochs = 15
	learning_rate = 0.01
	batch_size_train = 20
	train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy = getDataSet()
	model = Net(w_regularize, 'ResNetFCN')
	trainNet(model, epochs, learning_rate, batch_size_train, train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy)
