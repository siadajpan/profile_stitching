import numpy as np
import cv2

def get_dark_cols(img):
	(rows, cols, ch) = img.shape
	
	for start_col in range(cols):
		if img[:, start_col, :].mean() > 5:
			break
	
	for end_col in range(cols):
		if img[:, -end_col, :].mean() > 5:
			break
	print 'start: %s, end: %s' %(start_col, end_col)
    
	if end_col == 0:
		end_col = cols + 1
	else:
		end_col = -end_col

	return start_col, end_col

def preprocess_images(imageA, imageB, ratio):

	# get rid of dark left and right
	(start_col, end_col) = get_dark_cols(imageA)
	imageA = imageA[:, start_col:end_col]
	imageB = imageB[:, start_col:end_col]
	
	# rotate the images
	(rows,cols, channels) = imageA.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
	imageA = cv2.warpAffine(imageA,M,(rows,cols))
	imageB = cv2.warpAffine(imageB,M,(cols,rows))

	#resize to speed up stitching
	new_sizes = (int(cols * ratio), int(rows * ratio))
	imageA_stitch = cv2.resize(imageA, new_sizes)
	imageB_stitch = cv2.resize(imageB, new_sizes)
	
	return (imageA, imageA_stitch, imageB, imageB_stitch)
	
def create_result_array(imageA, no_of_images, ratio):
	(imageA, a, b, c) = preprocess_images(imageA, imageA, ratio)
	result = np.zeros((imageA.shape[0], imageA.shape[1]*no_of_images, 3), dtype = 'uint8')
	#print 'height: ', result.shape[0]
	#print 'width: ', result.shape[1]
	return result