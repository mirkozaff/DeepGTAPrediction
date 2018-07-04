import pickle
import gzip
import cv2
import re
import os.path as path
from deepgtav.messages import frame2numpy

DATA_PATH = 'd:data/'

file = gzip.open('d:dataset_test.pz')
filename = 'data.txt'

dir = 'training'
#dir = 'validation'
#dir = 'test'

data = open(path.join(DATA_PATH, dir, filename), 'w')
images_counter = 0

while True:
	try:
		data_dict = pickle.load(file) # Iterates through pickle generator
		if(len(data_dict['vehicles']) > 0):
			#Extract frame 
			frame = data_dict['frame']
			image = frame2numpy(frame, (1920,1080))
			frames_path = path.join(DATA_PATH, dir, 'frames')
			filename = format(images_counter, '06') + '.jpg' #saves images as six digit integer .jpg
			cv2.imwrite(path.join(frames_path, filename), image)
			
			#Extract vehicles info
			data.write(filename + ', ')
			vehicles_info = str(data_dict['vehicles'])
			result = re.findall(r'[\d\.]{2,}', vehicles_info) #removes non numeric values
			#print(str(result) + '\n')
			count = 1 #counter of vehicle info
			for val in result[:-1]:
				if(count%19 == 0): #newline for new vehicle if all vehicle info has been extracted
					data.write(val + '\n' + filename + ', ')
					count = 1
				else:
					data.write(val + ', ')
					count += 1
			
			data.write(result[-1] + '\n')

			images_counter += 1
	except EOFError:
		break

print(images_counter)
