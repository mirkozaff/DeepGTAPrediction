from keras.models import Model
from keras.layers import Dense, Input, Dropout, Reshape, concatenate, ELU, BatchNormalization
from keras.constraints import maxnorm
from keras.applications import ResNet50
import tensorflow as tf


def SDPN(summary=False):
	"""
	Create and return Semantic-aware Dense Prediction Network.

	Parameters
	----------
	summary : bool
		If True, network summary is printed to stout.

	Returns
	-------
	model : keras Model
		Model of SDPN

	"""
	input_coords = Input(shape=(4,))
	input_crop = Input(shape=(224, 224, 3))

	# extract feature from image crop
	resnet = ResNet50(include_top=False, weights='imagenet')
	for layer in resnet.layers:  # set resnet as non-trainable
		layer.trainable = False

	crop_encoded = resnet(input_crop)  # shape of `crop_encoded` is 2018x1x1
	crop_encoded = Reshape(target_shape=(2048,))(crop_encoded)

	# encode input coordinates
	h = Dense(256)(input_coords)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(256,kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(256,kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)

	# merge feature vectors from crop and coords
	merged = concatenate([crop_encoded, h])

	# decoding into output distance
	h = Dense(1024)(merged)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(512, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(256, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(128, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(64, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)

	output_dist = Dense(1, activation = 'relu')(h) #relu to output distance >= 0

	model = Model(inputs=[input_coords, input_crop], outputs=output_dist)

	if summary:
		model.summary()

	return model
