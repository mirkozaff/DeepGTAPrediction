import numpy as np
import cv2
import os.path as path
import keras

def imagenet_mean_bgr(frame_bgr, op='subtract'):
	"""
	Add or subtract ImageNet mean pixel value from a given BGR frame.
	"""
	imagenet_mean_BGR = np.array([123.68, 116.779, 103.939])

	frame_bgr = np.float32(frame_bgr)

	for c in range(0, 3):
		if op == 'subtract': frame_bgr[:, :, c] -= imagenet_mean_BGR[c]
		elif op == 'add':    frame_bgr[:, :, c] += imagenet_mean_BGR[c]

	return frame_bgr

class EarlyStopping(keras.callbacks.EarlyStopping):
	"""
	Implements Keras EarlyStopping setting the start epoch of the callback
    """
	def __init__(self, monitor='val_loss',
			 min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 1): # add argument for starting epoch
		super(EarlyStopping, self).__init__()
		self.start_epoch = start_epoch

	def on_epoch_end(self, epoch, logs=None):
		if epoch > self.start_epoch:
			super().on_epoch_end(epoch, logs)


def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((1080, 1920, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
	#flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    
    return rgb_flow








