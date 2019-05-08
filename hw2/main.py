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
	#focal_len = readf(sys.argv[1])
	#features = np.load('features.npy')
	features = []
	descriptors = []
	warp_img = []
	for i,element in enumerate(img):
		element = warping(element)
		warp_img.append(element)
		feature = HarrisDetection(element)
		features.append(feature)
		descriptors.append( gen_descriptor(element, features[i]) )
	#features = np.save("features",features)
	feature_maps = feature_matching(warp_img, features, descriptors)
	
	print ("done feature_maps", [len(x) for x in feature_maps])
	result = warp_img[0]
	for i,feature in enumerate(feature_maps):
		h = ransac(feature, features[i], features[i+1])
		result = blending(result, warp_img[i+1], h)
		cv2.imwrite("1dst.jpg",result)
	height_0, height_1 = 0,0
	height, width = result.shape[:2]

	for i in range(height):
		if not np.any(result[i]==0):
			height_0 = i
			break
	for i in range(height-1,0,-1):
		if not np.any(result[i]==0):
			height_1 = i
			break
	crop = result[height_0:height_1,:]
	cv2.imwrite("crop.jpg",crop)
