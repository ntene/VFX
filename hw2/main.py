import cv2
import sys
from detection import *
from local_similarity import *
import os
from c_warp import *
#from warping import *
def readimg(path):
	img = []
	with open(os.path.join(path,'list'), 'r') as f:
		for line in f:
			img.append(cv2.imread(path+line.strip()))
	return img

def readf(path):
	focal_len = []
	with open(os.path.join(path,"focal_len"),'r') as f:
		for lines in f:
			focal_len.append(float(lines.strip()))
	return focal_len

if __name__ == '__main__':
	img = readimg(sys.argv[1])
	focal_len = readf(sys.argv[1])
	features = []
	descriptors = []
	warp_img = []
	for i,element in enumerate(img[0:2]):
		element = warping(element,focal_len[i])
		warp_img.append(element)
		feature = HarrisDetection(element)
		features.append(feature)
		descriptors.append( gen_descriptor(element, feature) )
	feature_maps = feature_matching(warp_img, features, descriptors)
	print ("done feature_maps", len(feature_maps[0]))
	result = warp_img[0]
	first_shift = 0
	last_shift = 0
	for i,feature in enumerate(feature_maps):
		h = ransac(feature, features[i], features[i+1])
		last_shift += h[0]
		result = blending(result, warp_img[i+1], h)
		cv2.imwrite("dst.jpg",result)
	print (first_shift, last_shift)
	crop = result[last_shift:result.shape[0]-last_shift,:]
	cv2.imwrite("dst1.jpg",crop)
		#img1_w = transform_p(warp_img[i], warp_img[i+1], h)
		#cv2.imwrite('ww.jpg', img1_w)

	#cv2_sift(img[0:2])
