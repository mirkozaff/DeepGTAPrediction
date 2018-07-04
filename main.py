import matplotlib
matplotlib.use('Agg')
from distance_model import SDPN
from speed_model import nvidia_model, ODFPA
from keras.optimizers import Adam
from keras.models import load_model
from load_batch import convert_from_relative_to_absolute, DistanceDataGenerator, load_distance_dataset, load_speed_dataset, SpeedDataGenerator
import tensorflow as tf
import numpy as np
import csv
import os.path as path
import cv2
from matplotlib import pyplot as plt
from ModelMGPU import ModelMGPU

D_MODEL_PATH = 'models/'
S_MODEL_PATH = 'speed_models/'
#DATA_PATH = 'd:data'
DATA_PATH = '/mnt/data1/zaffaroni_data/data'

if __name__ == '__main__':
	
	#Distance Prediction
	# Get model
	d_model = SDPN(summary=False)

	#Pre-trained model path
	d_pretrained_model_path = D_MODEL_PATH + 'model_checkpoint.h5'

	# Load pre-trained weights
	d_model.load_weights(d_pretrained_model_path)
	d_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='mse')

	#Setting variables
	batch_size = 1

	#Loading dataset
	s_test_dataset = load_speed_dataset(data_dir = 'test')
	d_test_dataset = []
	for item in s_test_dataset:
		#d_test_dataset.append(item[0])
		d_test_dataset.append(item[1])

	#Setting generators
	d_test_generator = DistanceDataGenerator(d_test_dataset, batch_size, shuffle=False)

	# Perform distance prediction given (vehicle_coords, vehicle_crop)
	print('Prediction started')
	Y_dist_pred = d_model.predict_generator(d_test_generator, verbose = 1)
	print('Prediction ended')
	
	#Extract distance prediction from array of prediction
	temp = []
	for y in Y_dist_pred:
		temp.append(y[0])	
	d_Y_prediction = np.array(temp)

	#Extract distance ground truth from data
	d_Y_real = []
	for y in d_test_dataset:
		d_Y_real.append(y[3])

	#Speed Prediction
	# Get model
	s_model = ODFPA(summary=False)

	#Pre-trained model path
	s_pretrained_model_path = S_MODEL_PATH + 'model_checkpoint.h5'

	# Load pre-trained weights
	s_model.load_weights(s_pretrained_model_path)
	s_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='mse')

	#Setting variables
	batch_size = 1

	#Setting generators
	s_test_generator = SpeedDataGenerator(s_test_dataset, batch_size, shuffle=False)

	# Perform speed prediction given dense optical flow
	print('Prediction started')
	Y_speed_pred = s_model.predict_generator(s_test_generator, verbose = 1)
	print('Prediction ended')
	
	#Extract speed prediction from array of prediction
	temp = []
	for y in Y_speed_pred:
		temp.append(y[0])	
	s_Y_prediction = np.array(temp)

	#Extract Speed ground truth from data
	s_Y_real = []
	for y in s_test_dataset:
		s_Y_real.append(y[1][2])
	
	# Save prediction on csv
	with open('prediction.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		spamwriter.writerow(('Distance Real','Distance Prediction', 'Speed Real', 'Speed Prediction'))
		for d_real,d_pred,s_real,s_pred in zip(d_Y_real, d_Y_prediction, s_Y_real, s_Y_prediction):
			spamwriter.writerow((str(d_real),str(d_pred),str(s_real),str(s_pred)))
	print('File prediction.csv created')
			
	logs = d_test_dataset[:]
	frame_prec = ''
	i = 0

	#Save images with bounding boxes and prediction labels
	for log in logs:	
		# parse a log line
		frame = log[0]
		entityID = log[1]
		speed, distance = log[2:4]
		points = log[4:]
		points = [float(i) for i in points]

		bb = list(convert_from_relative_to_absolute(1080, 1920, *points))

		if frame != frame_prec:
			if frame != 'zero':
				filename = (frame_prec.strip().split('.')[0]).strip().split('/')[-1] + '.png' #'\\' on windows
				plt.savefig(path.join(DATA_PATH, 'video', filename), bbox_inches='tight',transparent=True, pad_inches=0)
			
			image = path.join(frame)
			image = cv2.imread(image, cv2.IMREAD_COLOR)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			plt.close()
			plt.figure(figsize=(20,12))
			plt.imshow(image)

			current_axis = plt.gca()
			plt.axis('off')
		
		#Bounding box info
		xmin = bb[0]
		ymin = bb[1]
		xmax = bb[2]
		ymax = bb[3]

		color = 'red'
		label = '{}:{:.3f}\n{}:{:.3f}\n{}:{:.3f}\n{}:{:.3f}'.format(
			'dReal', float(distance), 'dPred', d_Y_prediction[i],'sReal', float(speed), 'sPred', s_Y_prediction[i])
		current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
		current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

		i += 1
		frame_prec = frame
		print(i)