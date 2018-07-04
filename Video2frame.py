import cv2
import os.path as path

#DATA_PATH = 'd:data'
DATA_PATH = '/mnt/data1/zaffaroni_data/data'
dir = 'video_real'

#video to extract
vidcap = cv2.VideoCapture(path.join(DATA_PATH, dir, 'GH010005.mp4'))

# Find the number of frames
video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
success,image = vidcap.read()
count = 0
while count < (video_length-1):
	#if the frame was extracted with success, save it
	if(success):	
		cv2.imwrite(path.join(DATA_PATH, dir, 'frames', "frame_%d.jpg" % count), image)     # save frame as JPEG file      
	success,image = vidcap.read()
	count += 1
