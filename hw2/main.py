import cv2
import sys
from detection import *
from local_similarity import *
import os
#from c_warp import *
from warping import *
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

	for i,feature in enumerate(feature_maps):
		h = ransac(feature, features[i+1], features[i])
		height, width, _ = img[i].shape
		img1_w = transform_p(warp_img[i], warp_img[i+1], h)
		#cv2.imwrite('ww.jpg', img1_w)

	#cv2_sift(img[0:2])