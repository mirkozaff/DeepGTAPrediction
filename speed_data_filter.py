import os.path as path
import numpy as np

DATA_PATH = 'd:data/'

filename = 'filtered_data.txt'
filename2 = 'speed_filtered_data.txt'

dir = 'training'
#dir = 'validation'
#dir = 'test'

data = open(path.join(DATA_PATH, dir, filename2), 'w')
sample_counter = 0

#extract subsequent frames from the distance dataset
with open(path.join(DATA_PATH, dir, filename), 'r') as f:
	logs = f.readlines()
	logs2 = logs[:]

	for log in logs:

		# retrieve line values
		line = log
		log = log.strip().split(',')

		# parse a log line
		frame = log[0]
		entityID = log[1]
		speed, distance = log[2:4]
		bbox3D_X = list(map(np.float32,log[4::2]))
		bbox3D_Y = list(map(np.float32,log[5::2]))

		#get 2D bounding box from 3D bounding box
		min_x = min(bbox3D_X)
		max_x = max(bbox3D_X)
		min_y = min(bbox3D_Y)
		max_y = max(bbox3D_Y)

		for log2 in logs2:
			# retrieve line values
			log2 = log2.strip().split(',')

			frame2 = log2[0]
			entityID2 = log2[1]
			frame_next = int(frame2.strip().split('.')[0])
			frame_prec = int(frame.strip().split('.')[0])
			#if there is no frame next then break 
			if(frame_next >= frame_prec + 2):
				break
			#if there is frame next then parse and save it
			if((frame_next == frame_prec + 1) and (entityID2 == entityID)):				
				speed2, distance2 = log2[2:4]
				bbox3D_X2 = list(map(np.float32,log2[4::2]))
				bbox3D_Y2 = list(map(np.float32,log2[5::2]))

				#get 2D bounding box from 3D bounding box
				min_x2 = min(bbox3D_X2)
				max_x2 = max(bbox3D_X2)
				min_y2 = min(bbox3D_Y2)
				max_y2 = max(bbox3D_Y2)

				data.write(frame + ',' + entityID + ',' + speed + ',' + distance + ',' + str(min_x) + ',' + str(min_y) + ',' + str(max_x) + ',' + str(max_y) + '\n')
				data.write(frame2 + ',' + entityID2 + ',' + speed2 + ',' + distance2 + ',' + str(min_x2) + ',' + str(min_y2) + ',' + str(max_x2) + ',' + str(max_y2) + '\n')

				logs2 = logs2[logs2.index(line):len(logs2)]
				sample_counter += 1
				break

print(sample_counter)		
