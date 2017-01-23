import numpy as np
from keras import backend as K

def dice_coeff(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred, smooth = 1):
    return 1-dice_coeff(y_true, y_pred)

def np_dice_coeff(y_true, y_pred, smooth = 1):
	y_true_f = y_true.flatten()
	#np.savetxt('Val_true.txt', y_true_f) #Debug
	y_pred_f = y_pred.flatten()
	#np.savetxt('Val_pred.txt', y_pred_f) #Debug
	intersection = np.sum(y_true_f * y_pred_f)
	return ( (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
