from keras.models import Sequential, Model
from keras.constraints import maxnorm
from keras.layers import ELU, Conv2D, Dense, Dropout, Lambda, Flatten, BatchNormalization, Input, concatenate
import numpy as np
import tensorflow as tf

#images size
N_img_height = 66
N_img_width = 220
N_img_channels = 3

def nvidia_model(summary=False):
	inputShape = (N_img_height, N_img_width, N_img_channels)

	model = Sequential()

	#Normalization
	model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

	model.add(Conv2D(24, (5, 5), name="conv1", strides=(2, 2), padding="valid", kernel_initializer="he_normal"))	
	model.add(ELU())
	model.add(BatchNormalization())

	model.add(Conv2D(36, (5, 5), name="conv2", strides=(2, 2), padding="valid", kernel_initializer="he_normal"))	
	model.add(ELU()) 
	model.add(BatchNormalization())

	model.add(Conv2D(48, (5, 5), name="conv3", strides=(2, 2), padding="valid", kernel_initializer="he_normal"))
	model.add(ELU())
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Conv2D(64, (3, 3), name="conv4", strides=(1, 1), padding="valid", kernel_initializer="he_normal"))	
	model.add(ELU())  
	model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3), name="conv5", strides=(1, 1), padding="valid", kernel_initializer="he_normal"))
	model.add(ELU())
	model.add(BatchNormalization())

	model.add(Flatten(name = 'flatten'))
	model.add(ELU())
	model.add(Dropout(0.25))

	model.add(Dense(100, name="fc1", kernel_initializer="he_normal"))
	model.add(ELU())
	model.add(Dropout(0.25))

	model.add(Dense(50, name="fc2", kernel_initializer="he_normal"))
	model.add(ELU())
	model.add(Dropout(0.25))

	model.add(Dense(10, name="fc3", kernel_initializer="he_normal"))
	model.add(ELU())
	model.add(Dropout(0.25))
	
	# do not put activation at the end because we want to exact output, not a class identifier
	model.add(Dense(1, name="output", kernel_initializer="he_normal")) 

	if summary:
		model.summary()

	return model

def ODFPA(summary=False):

	input_coords = Input(shape=(8,))
	input_crop = Input(shape=(N_img_height, N_img_width, N_img_channels))

	# extract feature from image optical flow
	pilotNet = nvidia_model()

	speed_encoded = pilotNet(input_crop)

	# encode input coordinates
	h = Dense(256)(input_coords)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(256, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)
	h = Dropout(rate=0.4)(h)
	h = Dense(256, kernel_constraint=maxnorm(3))(h)
	h = ELU()(h)

	# merge feature vectors from crop and coords
	merged = concatenate([speed_encoded, h])

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

	output_speed = Dense(1, activation = 'relu')(h)
	
	model = Model(inputs=[input_coords, input_crop], outputs=output_speed)

	if summary:
		model.summary()

	return model

