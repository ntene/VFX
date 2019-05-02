import cv2
import sys
from detection import *
from local_similarity import *

def readimg(path):
	img = []
	with open(path+'list', 'r') as f:
		for line in f:
			img.append(cv2.imread(path+line.strip()))
	return img

if __name__ == '__main__':
	img = readimg(sys.argv[1])
	features = []
	descriptors = []
	for element in img:
		feature = HarrisDetection(element)
		features.append(feature)
		descriptors.append( gen_descriptor(element, feature) )
	
	feature_maps = feature_matching(img, features, descriptors)

