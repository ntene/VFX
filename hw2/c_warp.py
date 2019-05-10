import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
def warping(img):
	f = 917.4311926605504
	img = cv2.resize(img,(630,1120))
	h, w, _ = img.shape
	x_0 = round(f*math.atan((0-w/2)/f) + w/2)
	xf = round(f*math.atan((w-w/2)/f) + w/2)
	y_0 = round(f*((0-h/2)/math.sqrt(math.pow((0-w/2),2) + math.pow(f,2))) + h/2)
	w_img = np.zeros((h,w,3), dtype=np.uint8)
	for x in range(w):
		for y in range(h):
			x_ = round(f*math.atan((x-w/2)/f) + w/2)
			y_ = round(f*((y-h/2)/math.sqrt(math.pow((x-w/2),2) + math.pow(f,2))) + h/2)
			if x_ < 0 or y_ < 0 or x_ >= xf-x_0 or y_ >= h:
				pass
			else:
				w_img[y_][x_][:] = img[y][x][:]
	return w_img

def ransac(match_features, feature1, feature2):
	best_h = []
	max_inliers = 0
	match_pairs = []
	best_inliers = []
	for i in range(len(match_features)):
		u = feature1[match_features[i][0]]
		v = feature2[match_features[i][1]]
		match_pairs.append([u,v])
	match_pairs = np.array(match_pairs)

	for match in match_pairs:
		shift = match[1] - match[0]

		v_ = match_pairs[:,1] - shift
		dist = match_pairs[:,0] - v_
		inliers = 0
		for j in range(dist.shape[0]):
			if np.sum(np.square(dist[j])) < 5:
				inliers += 1
		if inliers > max_inliers:
			best_h = shift
			max_inliers = inliers
	print (best_h, max_inliers)
	return best_h

def blending(img1, img2, shift):
	x_shift = shift[1]
	y_shift = shift[0]
	shift[0] = img2.shape[0] - abs(shift[0])
	shift[1] = img2.shape[1] - abs(shift[1])
	result_h = img1.shape[0] + img2.shape[0] - abs(shift[0])
	result_w = img1.shape[1] + img2.shape[1] - abs(shift[1])

	im_2_h, im_2_w,_ = img2.shape
	im_1_h, im_1_w,_ = img1.shape
	result_image = np.zeros((result_h,result_w,3))
	result_image[result_h - im_1_h: result_h, result_w - im_1_w: result_w] = 0.9*img1
	result_image[0:im_2_h,0:im_2_w] = 0.9*img2


	centre = int((im_2_w + result_w - im_1_w )/2) 
	start = result_w - im_1_w #img1
	end = im_2_w
	
	#start = end - 10
	'''for x in range(start, end):
		for y in range(0,im_2_h):
			result_image[y][x] = 0.8*result_image[y][x]'''

	start = int((centre + end) /2)

	for x in range(start, end):
		for y in range(0,im_2_h):
			result_image[y][x] = ((end-x)/(end-start))*img2[y][x] + ((x-start)/(end-start))*img1[y-result_h + im_1_h][x-result_w + im_1_w]

	return result_image
		


