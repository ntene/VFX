import numpy as np
import cv2
from cv2 import bilateralFilter

def tone_mapping(hdr, gamma=0.6, scale=5/255):
	Iweight = np.array([30.0, 40.0, 1.0]) / 71
	I = hdr[:,:,0] * Iweight[0] + hdr[:,:,1] * Iweight[1] + hdr[:,:,2] * Iweight[2]
	I[I==0] = 1e-4
	L = np.log2(I)
	base = bilateralFilter(L, d=-1, sigmaColor=5, sigmaSpace=5, borderType=cv2.BORDER_REPLICATE)
	detail = L - base
	offset = np.max(base)
	B_ = (base - offset) * scale
	O = pow(2, B_ + detail)
	RGB = [hdr[:,:,i] / I for i in range(3)]
	RGB_ = pow(O * RGB, 1/gamma).transpose(1,2,0)
	RGB_ *= 105
	return RGB_

