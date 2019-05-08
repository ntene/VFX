from scipy import signal as sig
import cv2
import numpy as np
import math

def gradient(img):
    kernel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(img, kernel_x, mode='same'), sig.convolve2d(img, kernel_y, mode='same')

def HarrisDetection(img, blocksize=4, k=0.05):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# sample show
	dest = cv2.cornerHarris(gray, blocksize, 5, k)
	dest = cv2.dilate(dest, None)
	image = img.copy()
	image[dest > 0.4 * dest.max()]=[0, 0, 255]
	keylist = np.array(np.where(dest > 0.4)).T
	print (keylist.shape)
	cv2.imwrite('sample.jpg',image)
	return keylist
	###

	dx, dy = gradient(gray)
	Ixx = dx**2
	Ixy = dx*dy
	Iyy = dy**2
	height, width = gray.shape
	offset = blocksize // 2
	R = np.zeros((height, width))
	keylist = []

	for y in range(1+offset, height-offset-1, 2):
		for x in range(1+offset, width-offset-1, 2):
			Wxx = np.sum( Ixx[y-offset:y+offset+1, x-offset:x+offset+1] )
			Wxy = np.sum( Ixy[y-offset:y+offset+1, x-offset:x+offset+1] )
			Wyy = np.sum( Iyy[y-offset:y+offset+1, x-offset:x+offset+1] )

			det = (Wxx * Wyy) - (Wxy**2)
			trace = Wxx + Wyy
			R[y][x] = det - k * (trace**2)
			keylist.append([y,x,R[y][x]])
	
	keylist.sort(key=lambda x: x[2], reverse=True)
	keylist = np.array(keylist)[:2000, :2].astype('int')
	
	# show
	image = img.copy()
	for key in keylist:
		image[key[0]][key[1]] = [0,0,255]
	cv2.imwrite('corner.jpg', image)
	###

	return keylist
				
def gen_descriptor(img, keylist):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	descriptors = []
	for key in keylist:
		descriptor = []
		for y in range(-2,3):
			for x in range(-2,3):
				descriptor.append(gray[key[0]+y][key[1]+x])
		descriptors.append(descriptor)
	return np.array(descriptors)	

