import os.path as path
import numpy as np

DATA_PATH = 'd:data/'

filename = 'data.txt'
filename2 = 'filtered_data2.txt'

dir = 'training'
#dir = 'validation'
#dir = 'test'

data = open(path.join(DATA_PATH, dir, filename2), 'w')
sample_counter = 0

with open(path.join(DATA_PATH, dir, filename), 'r') as f:
	logs = f.readlines()

	for log in logs:

		# retrieve line values
		log = log.strip().split(',')
		if(len(log) == 20):
			# parse a log line
			frame = log[0]
			entityID = log[1]
			speed, distance = log[2:4]
			bbox3D_X = list(map(np.float32,log[4::2]))
			bbox3D_Y = list(map(np.float32,log[5::2]))

			# Get 2D bounding box from 3D bounding box
			min_x = min(bbox3D_X)
			max_x = max(bbox3D_X)
			min_y = min(bbox3D_Y)
			max_y = max(bbox3D_Y)
	
			# Round and convert values
			distance = "{:.3f}".format(float(distance)/150)
			speed = "{:.3f}".format(float(speed))			

			# Remove incosistent line
			if ' 1.0' not in log[4:20]:
				data.write(frame + ',' + entityID + ',' + speed + ',' + distance + ',' + str(min_x) + ',' + str(min_y) + ',' + str(max_x) + ',' + str(max_y) + '\n')
				sample_counter += 1

print(sample_counter)		
