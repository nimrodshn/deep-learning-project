import numpy as np
from keras import backend as K
import tensorflow as tf

def dice_coeff(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred):
    dc_value_tenzor = dice_coeff(y_true, y_pred)
    const_value_tensor = tf.constant(1, tf.float32)
    return tf.sub(const_value_tensor, dc_value_tenzor)

def np_dice_coeff(y_true, y_pred, smooth = 1):
	y_true_f = y_true.flatten()
	
	y_pred_f = y_pred.flatten()

	intersection = np.sum(y_true_f * y_pred_f)
	return ( (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
