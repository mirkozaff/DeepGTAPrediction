import threading
import numpy as np
import cv2
import os
import os.path as path
from random import shuffle
from utils import imagenet_mean_bgr, change_brightness, opticalFlowDense
import keras

#DATA_PATH = 'd:data'
DATA_PATH = '/mnt/data1/zaffaroni_data/data'


def convert_from_relative_to_absolute(h, w, x_min, y_min, x_max, y_max):
	"""
	Convert from relative coordinates (range 0, 1) to absolute coordinates given a frame (range h, w)

	Parameters
	----------
	h : int
		Image height
	w : int
		Image width
	x_min : float
		X coordinate of top-left corner (in range 0, 1)
	y_min : float
		Y coordinate of top-left corner (in range 0, 1)
	x_max : float
		X coordinate of bottom-right corner (in range 0, 1)
	y_max : float
		Y coordinate of bottom-right corner (in range 0, 1)

	Returns
	-------
	coords : list
		Input coordinates casted to image size -> range (0, h) and (0, w)
	"""
	x_min = x_min * w
	y_min = y_min * h
	x_max = x_max * w
	y_max = y_max * h
	return map(np.int32, [x_min, y_min, x_max, y_max])


def extract_crop(frame, x_min, y_min, x_max, y_max):
	"""
	Extract vehicle crop from the image.
	Crop is resized to 224x224 which is ResNet input size.

	Parameters
	----------
	frame : ndarray
		Image to process
	x_min : float
		X coordinate of top-left corner (in range 0, 1)
	y_min : float
		Y coordinate of top-left corner (in range 0, 1)
	x_max : float
		X coordinate of bottom-right corner (in range 0, 1)
	y_max : float
		Y coordinate of bottom-right corner (in range 0, 1)

	Returns
	-------
	crop : ndarray
		Crop containing vehicle, resized to 224x224 pixel
	"""
	h, w = frame.shape[:2]

	x_min, y_min, x_max, y_max = convert_from_relative_to_absolute(h, w, x_min, y_min, x_max, y_max)

	# extract crop from frame
	crop = frame[y_min:y_max, x_min:x_max, :].copy()

	crop = cv2.resize(crop, (224, 224))

	return crop

def speed_extract_crop(frame, x_min, y_min, x_max, y_max):
	"""
	Extract vehicle crop from the image.
	Crop is resized to 220x66 which is PilotNet input size.

	Parameters
	----------
	frame : ndarray
		Image to process
	x_min : float
		X coordinate of top-left corner (in range 0, 1)
	y_min : float
		Y coordinate of top-left corner (in range 0, 1)
	x_max : float
		X coordinate of bottom-right corner (in range 0, 1)
	y_max : float
		Y coordinate of bottom-right corner (in range 0, 1)

	Returns
	-------
	crop : ndarray
		Crop containing vehicle, resized to 220x66 pixel
	"""
	h, w = frame.shape[:2]

	x_min, y_min, x_max, y_max = convert_from_relative_to_absolute(h, w, x_min, y_min, x_max, y_max)

	crop_h = y_max-y_min
	crop_w = x_max-x_min

	# extract crop from frame
	crop = frame[max(0, y_min-(2*crop_h)):min(y_max+(2*crop_h), h), :, :].copy()

	crop = cv2.resize(crop, (220, 66), interpolation = cv2.INTER_AREA)

	return crop

def subdirList(dir):
	"""
	Get a list of all sub directory in a directory. It is not recursive.

	Parameters
	----------
	dir: string
		the directory of interest

	Returns
	-------
	subdir : list
		list of all sub directory in a directory
	"""
	return [subdir for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]

class DistanceDataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, logs, batch_size, shuffle=True):
		'Initialization'
		self.logs = logs
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.logs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		logs_temp = [self.logs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(logs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.logs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, logs_temp):
		'Generates data containing batch_size samples' 
		#Initialization
		X_coords, X_crops, Y_dist = [], [], []

		for log in logs_temp:  

			# parse a log line
			frame_path = log[0]
			bbox_dist = log[3]
			bbpoints = list(map(np.float32, log[4:8]))

			# load images
			frame_path = path.join(frame_path)
			if not path.exists(frame_path): continue
			image = cv2.imread(frame_path, cv2.IMREAD_COLOR)

			# extract crops from whole frames
			crop = extract_crop(image, *bbpoints)
					
			if crop is not None:                        
				# append all needed stuff to output structures
				X_coords.append(bbpoints)  # append frontal coords
				X_crops.append(crop)  # append frontal crops
				Y_dist.append(bbox_dist)  # append bbox distance
							
		# preprocess X crops by subtracting mean
		for b in range(0, len(X_coords)):
			X_crops[b] = imagenet_mean_bgr(frame_bgr=X_crops[b], op='subtract')

		# convert all data to ndarray
		X_coords = np.array(X_coords)
		X_crops = np.array(X_crops)
		Y_dist = np.array(Y_dist)
					
		return [X_coords, X_crops], Y_dist

def load_distance_dataset(data_dir):
	"""
	Load distance info from a dataset saved as text file.

	Parameters
	----------
	data_dir: string
		folder of interest tipically from ['train','validation','test']

	Returns
	-------
	data : list
		list of alla data inside the dataset, every data is a list too
	"""
	data = []

	#look for all sub directory in data_dir
	for subdir in subdirList(path.join(DATA_PATH, data_dir)):
		with open(path.join(DATA_PATH, data_dir, subdir, 'filtered_data.txt'), 'r') as f:
			logs = f.readlines()
			for log in logs:
				log = path.join(DATA_PATH, data_dir, subdir,'frames','') + log
				log = log.strip().split(',')
				data.append(log) 	#save log info as a list in a list
		f.closed 

	return data

class SpeedDataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, logs, batch_size, shuffle=True):
		'Initialization'
		self.logs = logs
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.logs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		logs_temp = [self.logs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(logs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.logs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, logs_temp):
		'Generates data containing batch_size samples' 
		#Initialization
		X_img, Y_speed, X_coords = [], [], []

		for log in logs_temp:		   
			# retrieve line values
			log1 = log[0]
						
			# parse a log line
			frame_path = log1[0]
			entity = log1[1]
			speed = float(log1[2])
			bbpoints = list(map(np.float32, log1[4:8]))

			log2 = log[1]
						
			# parse a log line of next frame
			frame_next_path = log2[0]
			entity_next = log2[1]
			speed_next = float(log2[2])
			bbpoints_next = list(map(np.float32, log2[4:8]))

			if(entity == entity_next):
				# load image
				frame_path = path.join(frame_path)
				if not path.exists(frame_path): continue
				image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				# load next image
				frame_next_path = path.join(frame_next_path)
				if not path.exists(frame_path): continue
				image_next = cv2.imread(frame_next_path, cv2.IMREAD_COLOR)
				image_next = cv2.cvtColor(image_next, cv2.COLOR_BGR2RGB)

				if('training' in frame_path):
					# Generate a random bright factor to apply to both images
					bright_factor = 0.2 + np.random.uniform()
					image = change_brightness(image, bright_factor)
					image_next = change_brightness(image_next, bright_factor)

				# extract crops from whole frames
				crop = speed_extract_crop(image, *bbpoints)
				crop_next = speed_extract_crop(image_next, *bbpoints_next)

				# compute optical flow send in images as RGB
				optfd = opticalFlowDense(crop, crop_next)
	
				if optfd is not None:                        
					# append all needed stuff to output structures
					X_coords.append(bbpoints+bbpoints_next)  # append frontal coords
					X_img.append(optfd) 
					Y_speed.append(speed_next) 
					
		# convert all data to ndarray
		X_coords = np.array(X_coords)
		X_img = np.array(X_img)
		Y_speed = np.array(Y_speed)
					
		return [X_coords, X_img], Y_speed

def load_speed_dataset(data_dir):	
	"""
	Load speed info from a dataset saved as text file.

	Parameters
	----------
	data_dir: string
		folder of interest tipically from ['train','validation','test']

	Returns
	-------
	data : list
		list of alla data inside the dataset, every data is a list too 
	"""
	data = []

	#look for all sub directory in data_dir
	for subdir in subdirList(path.join(DATA_PATH, data_dir)):
		with open(path.join(DATA_PATH, data_dir, subdir, 'speed_filtered_data.txt'), 'r') as f:
			#read two subsequent
			log = f.readline()
			log2 = f.readline()
			while log2:
				log = path.join(DATA_PATH, data_dir, subdir,'frames','') + log
				log = log.strip().split(',')
				log2 = path.join(DATA_PATH, data_dir, subdir,'frames','') + log2
				log2 = log2.strip().split(',')
				data.append([log, log2]) #save two subsequent frame in the same list
				log = f.readline()
				log2 = f.readline()
		f.closed

	return data