import cv2
import numpy as np
import skimage.feature
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt

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
	for i in range(len(hist_a)):
		if query_result_distance[i] < 250 and dot_set_a[i,1] < width/1.5 and dot_set_b[query_result_idx[i],1] > width/1.5:
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

	return np.array(feature_maps)

def cv2_sift(imgs):
	N = len(imgs) - 1
	for i in range(N):
		gray = cv2.cvtColor(imgs[i+1], cv2.COLOR_BGR2GRAY)
		gray1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
		orb = cv2.ORB_create(1000) 
		kp1, des1 = orb.detectAndCompute(gray,None)
		kp2, des2 = orb.detectAndCompute(gray1,None)	
		bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
		matches = bf.match(des1,des2,None)
		matches.sort(key=lambda x: x.distance, reverse=False)
		numGoodMatches = int(len(matches) * 0.15)
		matches = matches[:numGoodMatches]
		
		img3 = cv2.drawMatches(imgs[i+1],kp1,imgs[i],kp2,matches,None, flags=2)
		points1 = np.zeros((len(matches), 2), dtype=np.float32)
		points2 = np.zeros((len(matches), 2), dtype=np.float32)
		for j, match in enumerate(matches):
			points1[j, :] = kp1[match.queryIdx].pt
			points2[j, :] = kp2[match.trainIdx].pt
		h = test_ransac()
		#h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
		#result = l_warpTwoImages(imgs[i+1],imgs[i],h)
		#plt.imshow(result),plt.show()

def l_warpTwoImages(img2, img1, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    print ("h, pts2", H, pts2)
    pts = np.concatenate((pts1, pts2_), axis=0)
    print ("lpts",pts)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    print (Ht)
    print ("H", H)
    print ("dot", Ht.dot(H))
    print ("np.dot", np.dot(Ht,H))
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    print (xmax-xmin, ymax-ymin)
    cv2.imwrite("1p.jpg",result)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    cv2.imwrite("1p.jpg",result)
    return result
