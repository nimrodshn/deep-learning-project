from keras import backend as K
import numpy as np

def binary_cross_entropy(y_true, y_pred):
	true_f = K.flatten(y_true)
	pred_f = K.flatten(y_pred)
	return K.mean(true_f*K.log(pred_f) + (1-true_f)*K.log(pred_f))

def np_binary_cross_entropy(y_true, y_pred):
	true_f = y_true.flatten()
	pred_f = y_pred.flatten()
	return np.mean(true_f*np.log(pred_f))
