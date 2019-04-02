import numpy as np
import cv2

def MTB(images, layer, img_N, noise=4):
	if layer > 0:
		shrink_img = [cv2.resize(images[i], None, fx=1/2, fy=1/2) for i in range(img_N)]
		shiftX, shiftY = MTB(shrink_img, layer-1, img_N)
		shiftX *= 2
		shiftY *= 2
	else:
		shiftX = [0 for i in range(img_N)]
		shiftY = [0 for i in range(img_N)]
	
	median = [np.median(images[i]) for i in range(img_N)]
	binary_img = [cv2.threshold(images[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(img_N)]
	mask_img = [cv2.inRange(images[i], median[i]-noise, median[i]+noise) for i in range(img_N)]
	min_err = [float('Inf') for i in range(img_N)] 
	rX = [0 for i in range(img_N)]
	rY = [0 for i in range(img_N)]
	rows, cols = binary_img[0].shape

	for x in [-1, 0, 1]:
		for y in [-1, 0, 1]:
			for i in range(1,img_N):
				xs = shiftX[i] + x
				ys = shiftY[i] + y
				wrap_m = np.array([[1,0,xs],[0,1,ys]],dtype=np.float32)
				shift_binary = cv2.warpAffine(binary_img[i], wrap_m, (cols, rows))
				shift_mask = cv2.warpAffine(mask_img[i], wrap_m, (cols, rows))
				diff = np.bitwise_xor(binary_img[0], shift_binary)
				diff = np.bitwise_and(mask_img[0], diff)
				diff = np.bitwise_and(shift_mask, diff)
				err = np.sum(diff)
				if err < min_err[i]:
					rX[i] = xs
					rY[i] = ys
					min_err[i] = err
	return rX, rY

def Alignment(images, shiftX, shiftY):
	print (shiftX, shiftY)
	N = len(images)
	rows, cols, _ = images[0].shape
	images = [cv2.warpAffine(images[i], np.array([[1,0,shiftX[i-1]],[0,1,shiftY[i-1]]],dtype=np.float32), (cols, rows)) for i in range(N)]
	return images

def align(images, layer, num_images):
	imgs = [images[i][:,:,1] for i in range(num_images)]
	shiftX, shiftY = MTB(imgs, 3, num_images)
	#shiftX = [0, 0, -4, -4, -3, -1, 4, -4, -4, 4, 2, -4, -4, -4]
	#shiftY = [0, 0, -4, -4, -4, -4, -4, -4, 4, 4, -4, -4, -4, 4]
	images = Alignment(images, shiftX, shiftY)
	return images

