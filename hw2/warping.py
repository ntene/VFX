import math
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
def dlt(f, t, num_points=4):
	num_points = f.shape[0]
	A = np.zeros((2*num_points, 9))
	for p in range(num_points):
		fh = np.array([f[p,0], f[p,1], 1])										# Homogenous coordinate of point p
		A[2*p] = np.concatenate(([0, 0, 0], -fh, t[p,1]*fh))					# [0' -wX' yX']
		A[2*p + 1] = np.concatenate((fh, [0, 0, 0], -t[p,0]*fh))				# [wX' 0' -xX']
	U, D, V = np.linalg.svd(A)
	H = V[8].reshape(3, 3)
	return H / H[-1,-1]

def test_ransac(match_features, img1, img2, threshold_distance=0.5, threshold_inliers=0, ransac_iters=50):
	mac_1 = []
	mac_2 = []
	h = []
	for i in range(len(match_features)):
		mac_1.append(img1[match_features[i][0]])
		mac_2.append(img2[match_features[i][1]])
	for i in range(100):
		random.seed()
		index = [random.randint(0, len(match_features)-1) for i in range(4)]
		fp = np.array([mac_2[pt] for pt in index])
		tp = np.array([mac_1[pt] for pt in index])
		homography = dlt(fp, tp)        # tp = H*fp
		src_pts = np.insert(mac_2, 2, 1, axis=1).T # Add column of 1 at the end (Homogenous coordinates)
		dst_pts = np.insert(mac_1, 2, 1, axis=1).T # Add column of 1 at the end (Homogenous coordinates)
		projected_pts = np.dot(homography, src_pts)
		error = np.sqrt(np.sum(np.square(dst_pts - (projected_pts/projected_pts[-1])), axis=0))
		print (error)
		if np.count_nonzero(error < threshold_distance) > threshold_inliers:
			h = homography
			threshold_inliers = np.count_nonzero(error < threshold_distance)
	print (threshold_inliers)
	return h


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
	h_cv = np.array([[ 1.14932185e+00, -1.64356828e-03, -2.83512879e+02],
 [-1.40789302e-02,  1.29742331e+00, -2.03349509e+01],
 [ 4.03856448e-04 , 4.29683180e-04 , 1.00000000e+00]])
	for i in range(100):
		temp = []
		m1 = []
		m2 = []
		for j in range(4):
			random.seed()
			index = random.randint(0, len(match_features)-1)
			temp.append([img1[match_features[index][0]], img2[match_features[index][1]]])
			m1.append([img1[match_features[index][0]][1],img1[match_features[j][0]][0]])
			m2.append([img2[match_features[j][0]][1], img2[match_features[j][0]][0]])


		'''h = homography(temp)
		h = np.reshape(h,(3,3))
		h = h/h[-1,-1]'''
		h = dlt(np.array(m2),np.array(m1))

		inlier = 0
		for j in range(len(match_features) ):
			u = np.transpose(np.array([img1[match_features[j][0]][1], img1[match_features[j][0]][0],1]))
			v = np.transpose(np.array([img2[match_features[j][0]][1], img2[match_features[j][0]][0],1]))
			a = np.dot(h,v)
			b = np.dot(h_cv,v)
			bb = u-(b/b[-1])
			diff = math.sqrt(bb[0]**2 + bb[1]**2)
			aa = u-(a/a[-1])
			dist = math.sqrt(aa[0]**2 + aa[1]**2)
			print (diff, dist)
			
			if dist < 5:
				inlier += 1

		if inlier > max_inliers:
			best_h = h
			max_inliers = inlier

		print (max_inliers)
	return best_h

def transform_p(img2, img1, h):
	'''h = np.array([[ 1.14932185e+00, -1.64356828e-03, -2.83512879e+02],
 [-1.40789302e-02,  1.29742331e+00, -2.03349509e+01],
 [ 4.03856448e-04 , 4.29683180e-04 , 1.00000000e+00]])'''
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
	print ("hree")
	dst = dst.astype(np.uint8)

	return dst

def homography(img):
	A = []
	for element in img:
		y_, x_, y, x = element[0][0], element[0][1], element[1][0], element[1][1]
		A.append([0, 0, 0, -x, -y, -1, y_*x, y_*y, y_])
		A.append([x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_])
	print (A)
	A = np.array(A)
	u, s, v = np.linalg.svd(A)
	return v[8]
	
