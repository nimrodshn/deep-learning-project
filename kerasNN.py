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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, CSVLogger, ModelCheckpoint 
from DiceMetric import dice_coeff, dice_coeff_loss
from binary_cross import binary_cross_entropy
from MyDataManipulator import plot_pairwise_train, plot_pairwise_val, read_train_data, read_val_data, read_test_data


def Net(net_type, w_regularize=5e-4):
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

	

def trainNet(model,epochs, lrate, batch_size_train, train_set_numpy,train_label_set_numpy,val_set_numpy,val_label_set_numpy, net_type):
	
	#create a folder for training session.
	output_dir = createModelTimestampFolder(net_type)
	log_file_path = os.path.join(output_dir, 'train_looger.csv')
	best_chekpoint_file_path = os.path.join(output_dir, 'best_model.hdf5')
	

	adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# sgd = SGD(lr=lrate, momentum=0.9, decay=0.0, nesterov=False)

	validation_callback = Val_Callback(val_data=(val_set_numpy,val_label_set_numpy),train_data=(train_set_numpy, train_label_set_numpy) , model=model)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto')
	csv_logger = CSVLogger(log_file_path, separator=',', append=False)
	checkpoint_writer = ModelCheckpoint(best_chekpoint_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callback_list = [reduce_lr, csv_logger, checkpoint_writer]

	print(model.summary())
	model.compile(loss=[dice_coeff_loss], optimizer=adam, metrics=[dice_coeff])
	model.fit(train_set_numpy, train_label_set_numpy, callbacks=callback_list , validation_data=(val_set_numpy,val_label_set_numpy), batch_size=batch_size_train, nb_epoch=epochs, verbose=1)
	


def getDataSet():

	train_set_numpy, train_label_set_numpy = read_train_data()
	val_set_numpy, val_label_set_numpy = read_val_data()

	return train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy

def runNet(net_type):
	epochs = 15
	learning_rate = 0.01
	batch_size_train = 20
	train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy = getDataSet()
	model = Net(net_type)
	trainNet(model, epochs, learning_rate, batch_size_train, train_set_numpy, train_label_set_numpy, val_set_numpy, val_label_set_numpy, net_type)

def evalModelOnValAndTest(net_type, model_chekpoint_folder_path):
	model = Net(net_type)
	model_chekpoint_file_path = os.path.join(model_chekpoint_folder_path, 'best_model.hdf5')
	model.load_weights(model_chekpoint_file_path)
	model.compile(loss=[dice_coeff_loss], optimizer='adam', metrics=[dice_coeff])


	val_set_numpy, val_label_set_numpy = read_val_data()
	test_set_numpy, test_label_set_numpy = read_test_data()
	dice_score_validation = model.evaluate(x = val_set_numpy , y = val_label_set_numpy, verbose = 0)
	dice_score_test = model.evaluate(x = test_set_numpy, y = test_label_set_numpy, verbose = 0)
	print('\nValidation Dice Score: {} Testing Dice Score: {}\n'.format(dice_score_validation[1], dice_score_test[1]))
	return dice_score_validation, dice_score_test


def createModelTimestampFolder(net_type_string):
	net_dir = os.path.join(os.getcwd(), net_type_string + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(net_dir)
	return net_dir