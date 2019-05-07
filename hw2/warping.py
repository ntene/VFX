import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
def warping(img, f):
	h, w, _ = img.shape
	x_0 = round(f*math.atan((0-w/2)/f) + w/2)
	y_0 = round(f*((0-h/2)/math.sqrt(math.pow((0-w/2),2) + math.pow(f,2))) + h/2)
	w_img = np.zeros((h,w,3), dtype=np.uint8)
	for x in range(w):
		for y in range(h):
			x_ = round(f*math.atan((x-w/2)/f) + w/2 - x_0)
			y_ = round(f*((y-h/2)/math.sqrt(math.pow((x-w/2),2) + math.pow(f,2))) + h/2 - y_0)
			if x_ < 0 or y_ < 0 or x_ >= w or y_ >= h:
				w_img[y_][x_][:] = 0
			else:
				w_img[y_][x_][:] = img[y][x][:]
	return w_img

def ransac(match_features, img1, img2):
	best_h = []
	max_inliers = 0
	for i in range(len(match_features)-3):
		temp = []
		for j in range(4):
			temp.append([img1[match_features[i+j][0]], img2[match_features[i+j][1]]])

		h = homography(temp)
		h = np.reshape(h,(3,3))
		h = h/h[-1,-1]

		inlier = 0
		for j in range(len(match_features) ):
			u = np.transpose(np.array([img1[match_features[j][0]][0], img1[match_features[j][0]][1],1]))
			v = np.transpose(np.array([img2[match_features[j][0]][0], img2[match_features[j][0]][1],1]))
			a = np.matmul(h,v)
			a = u-(a/a[-1])
			dist = a[0]**2 + a[1]**2
			
			if dist < 10:
				inlier += 1

		if inlier > max_inliers:
			best_h = h
			max_inliers = inlier

	return best_h

def transform_p(img2, img1, h):
	h = np.array([[ 1.14932185e+00, -1.64356828e-03, -2.83512879e+02],
 [-1.40789302e-02,  1.29742331e+00, -2.03349509e+01],
 [ 4.03856448e-04 , 4.29683180e-04 , 1.00000000e+00]])
	h1,w1, _ = img1.shape
	h2,w2, _ = img2.shape
	pts1 = np.float32([[0,0,1],[0,h1,1],[w1,h1,1],[w1,0,1]]).reshape(-1,1,3)
	pts2 = np.float32([[0,0,1],[0,h2,1],[w2,h2,1],[w2,0,1]]).reshape(-1,1,3)
	pts2_ = np.zeros((4,1,2))
	for i,element in enumerate(pts2):
		temp = np.dot(h,element[0].T)#[:,[0,1]]
		pts2_[i] = np.array([(temp/temp[-1])[:2]])
	pts = np.concatenate((pts1[:,:,[0,1]], pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	Ht = np.dot(np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]),h)
	dst = np.zeros((ymax-ymin,xmax-xmin,3))
	for x in range(w1):
		for y in range(h1):
			points = np.dot(np.linalg.inv(Ht),np.array([x,y,1]).T)
			x_, y_, _ = (points/points[2]).astype(int)
			if x_ < 0 or y_ < 0 or x_ >= w1 or y_ >= h1:
				dst[y][x][:] = 0
			else:
				dst[y][x] = img1[y_][x_]
	cv2.imwrite("be4.jpg",dst)
	dst[t[1]:h1+t[1],t[0]:w1+t[0]] = img2
	cv2.imwrite("dst.jpg",dst)

def homography(img):
	A = []
	for element in img:
		x_, y_, x, y = element[0][0], element[0][1], element[1][0], element[1][1]
		A.append([0, 0, 0, -x, -y, -1, y_*x, y_*y, y_])
		A.append([x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_])

	A = np.array(A)
	u, s, v = np.linalg.svd(A)
	return v[8]
	
