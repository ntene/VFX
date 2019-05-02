import cv2
import numpy as np
import skimage.feature
from sklearn.neighbors import KDTree

def softmax(x):
    x += np.finfo(float).eps
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def L1_loss(a,b):
	return np.linalg.norm(a-b,1)

def L2_loss(a,b):
	return np.linalg.norm(a-b,2)

def cosine(a,b):
	return 1- np.dot(a,b)/(np.linalg.norm(a,2)*np.linalg.norm(b,2))
def cosine2(a,b):
	a = np.transpose(a,(1,0,2))
	b = np.expand_dims(b,0)
	k = a.shape[0]
	d = a.shape[2]

	dot_value = np.sum(np.multiply(a,b),axis=2)
	a_norm = np.sqrt(np.sum(np.square(a),axis=2))
	b_norm = np.sqrt(np.sum(np.square(b),axis=2))
	sim = dot_value / (a_norm * b_norm)
	return 1- sim
def softmax2(x):
    x += np.finfo(float).eps
    exp_x = np.exp(x)
    softmax_x = np.expand_dims(np.sum(exp_x,axis=1),1)/exp_x
    softmax2_x = softmax_x/np.expand_dims(np.sum(softmax_x,axis=1),1)
    return softmax2_x 
def hist_intersection(a,b):
	return 1- np.sum(a*(a<b)+b*(b<=a))/min(np.sum(a),np.sum(b))

def query_KDT(a,b,imga,imgb):
	kdt_a, dot_set_a, hist_a = a
	kdt_b, dot_set_b, hist_b = b

	height, width, _ = imga.shape

	query_result_distance, query_result_idx = kdt_b.query(hist_a, k=1)
	
	feature_map = []
	for i in range(len(hist_b)):
		if query_result_distance[i] < 150 and dot_set_a[i,1] < width/2 and dot_set_b[query_result_idx[i],1] > width/2:
			feature_map.append([i, query_result_idx[i][0]])
	
	# show
	newimg = np.zeros((height, width*2, 3), np.uint8)
	newimg[:, :width] = imgb
	newimg[:, width:] = imga

	for fm in feature_map:
		pt_a = (int(dot_set_a[fm[0],1]+width), int(dot_set_a[fm[0],0]))
		pt_b = (int(dot_set_b[fm[1],1]), int(dot_set_b[fm[1],0]))
		cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
	cv2.imwrite('matches.jpg', newimg)
	###
	
	return feature_map

def feature_matching(imgs, features, descriptors):
	N = len(imgs)
	feats = []
	for i in range(N):
		kdt = KDTree(descriptors[i], metric='euclidean')
		feats.append((kdt, features[i], descriptors[i]))
	
	feature_maps = [0 for i in range(N-1)]
	for i in range(N-1):
		feature_maps[i] = query_KDT(feats[i], feats[i+1], imgs[i], imgs[i+1])
	
	return feature_maps
