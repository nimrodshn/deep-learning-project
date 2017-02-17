from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, Deconvolution2D, AtrousConvolution2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input, merge
from keras.regularizers import l2



def modelEncoderDecoder(w_regularize):
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
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
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

def modelFCN(w_regularize):
	# Input Image 64x64
	main_input = Input(shape=(64, 64,1))
	x = Convolution2D(8, 3, 3, init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(w_regularize))(main_input)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)

	x = Convolution2D(8, 6, 6, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	
	x = Convolution2D(1, 12, 12, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('sigmoid')(x)
	

	model = Model(input=[main_input], output=[x])
	print(model.summary())
	return model

def modelResNetFCN(w_regularize):
	# Input Image 64x64
	main_input = Input(shape=(64, 64,1))

	x = Convolution2D(8, 3, 3, init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(w_regularize))(main_input)
	x = BatchNormalization(mode=0, axis=1)(x)
	pre_merge = Activation('relu')(x)

	x = Convolution2D(8, 3, 3, init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(w_regularize))(pre_merge)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)

	x = Convolution2D(8, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	merged_output = merge([pre_merge, x], mode='sum')
	x = Activation('relu')(merged_output)
	
	x = Convolution2D(1, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)	
	final_ouput = Activation('sigmoid')(x)
	
	model = Model(input=[main_input], output=[final_ouput])
	print(model.summary())
	return model


def modelEncoderDecoderResNet(w_regularize):
	# Input Image 64x64
	main_input = Input(shape=(64, 64,1))
	x = Convolution2D(4, 3, 3 , init='he_normal', input_shape=(64, 64,1), border_mode='same', W_regularizer=l2(w_regularize))(main_input)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	pre_merge = MaxPooling2D(pool_size=(2, 2))(x)

	# Input Image 32x32
	x = Convolution2D(4, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(pre_merge)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	
	# Input Image 16x16
	x = Convolution2D(8, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# Input Image 8x8
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(8, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('relu')(x)
	# Inputs Image 16x16
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(4, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	merged_output = merge([pre_merge, x], mode='sum')
	x = Activation('relu')(merged_output)
	# Inputs Image 32x32
	x = UpSampling2D(size=(2,2))(x)
	x = AtrousConvolution2D(1, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(w_regularize))(x)
	x = BatchNormalization(mode=0, axis=1)(x)
	x = Activation('sigmoid')(x)
	

	model = Model(input=[main_input], output=[x])
	print(model.summary())
	return model