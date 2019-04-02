import cv2
import sys
import os
import numpy as np
import math
from fractions import Fraction
from ToneMapping import *
from MTB import *
import matplotlib.pyplot as plt
f = open(sys.argv[1]+"/list")
img_path = []
exposures = []
for lines in f:
    filename, exposure = lines.split()
    img_path.append(os.path.join(sys.argv[1],filename))
    exposures.append(float(Fraction(exposure)))

num_images = len(img_path)
images = [cv2.imread(i,1) for i in img_path ]
#images = align(images, 3, num_images)
exposures = np.array(exposures,dtype=np.float32)
height, width, _ = images[0].shape
print (np.array(images).shape)

img = np.array([images[i][j][k] for i in range(num_images) for j in range(0,height,250) for k in range(0,width,250)])
num_pixels, _ = img.shape
num_pixels = num_pixels//num_images
Z = img.reshape((num_images,num_pixels,3)).transpose(2,1,0)
print (Z.shape)
z_min = 0
z_max = 255

B = [math.log(i) for i in exposures]
l = 40
w_z = np.array([z if z <= 0.5*z_max else z_max - z for z in range(256)])

def gsolve(Z,B,l,w_z):
    n = 256
    N, M = Z.shape
    A = np.zeros((N*M+n+1, n+N),dtype=np.float32)
    b = np.zeros((A.shape[0],1),dtype=np.float32)

    k = 0
    for i in range(N):
        for j in range(M):
            wij = w_z[Z[i][j]]
            A[k][Z[i][j]+1] = wij
            A[k][n+i] = -wij
            b[k][0] = wij * B[j]
            k += 1

    A[k][128] = 1
    k += 1

    for i in range(n-1):
        A[k][i] = l*w_z[i+1]
        A[k][i+1] = -2 * l * w_z[i+1] 
        A[k][i+2] = l*w_z[i+1]
        k += 1


    x = np.linalg.lstsq(A, b,rcond=None)[0]
    g = x[0:n].flatten()
    lE = x[n+1:x.shape[0]]
    return g #,lE


g = np.array([gsolve(Z[i],B,l,w_z) for i in range(3)])
images = np.array(images).transpose(3,0,1,2)
Z = images.reshape((3,len(img_path),-1)).transpose(0,2,1)
_, N, M = Z.shape
ln_E = np.zeros((3, N))

E = []
print ("=====")
for k in range(3):
    de_num = np.sum(w_z[ [Z[k]] ] * ( g[k][ [Z[k]] ] - B ), axis=1)
    num = np.sum(w_z[ [Z[k]] ], axis=1)
    num[num<=0] = 1
    ln_E[k] = de_num/num
    E.append(np.array([np.exp(ln_E[k])]).reshape(height,width))

E = np.array(E,dtype=np.float32).transpose(1,2,0)
E_8bit = np.clip(E*255, 0, 255).astype('uint8')
im_color = cv2.applyColorMap(E_8bit, cv2.COLORMAP_HSV)
cv2.imwrite(sys.argv[1]+'aradiance.jpg', im_color)

tonemap1 = cv2.createTonemapDrago(1.0,0.7)
res_debvec = tonemap1.process(E.copy())
res_debvec = 3*res_debvec
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
cv2.imwrite(sys.argv[1]+"adrago.jpg", res_debvec_8bit)

tonemap1 = cv2.createTonemapDurand(1.5,4,1.0,1,1)
res_debvec = tonemap1.process(E.copy())
res_debvec = 3 * res_debvec
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
cv2.imwrite(sys.argv[1]+"adurand.jpg", res_debvec_8bit)

tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(E.copy())
ldrMantiuk = 3 * ldrMantiuk
cv2.imwrite(sys.argv[1]+"amantiuk.jpg", ldrMantiuk * 255)

# plot recovered response function
plt.plot(g[0],np.arange(256),'r.',g[1],np.arange(256),'g.',g[2],np.arange(256),'b.')
plt.ylabel('pixel value Z')
plt.xlabel('log exposure X')
plt.title('recovered response function')
plt.figure()

plt.imshow(im_color)
plt.show()

'''tonemap1 = cv2.createTonemapDrago(1.0,0.7)
res_debvec = tonemap1.process(E.copy())
res_debvec = 3*res_debvec
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
cv2.imwrite("hdr.jpg", res_debvec_8bit)

tonemap1 = cv2.createTonemapDurand(1.5,4,1.0,1,1)
res_debvec = tonemap1.process(E.copy())
res_debvec = 3 * res_debvec
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
cv2.imwrite("hd.jpg", res_debvec_8bit)

tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(E.copy())
ldrMantiuk = 3 * ldrMantiuk
cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)

l_remap = (0, 1)
saturation = 2.
numtiles = (4, 4)
ldrDrago = tonemap(E, l_remap=l_remap, saturation=saturation, numtiles=numtiles)
t = np.clip(ldrDrago*255, 0, 255).astype('uint8')
cv2.imwrite("_mytonemap_full.jpg", t[...,[2,1,0]] )

sample_hdr = cv2.imread(img_path[3], 1)
print (sample_hdr.shape)
im_color = cv2.applyColorMap(sample_hdr, cv2.COLORMAP_COOL)
cv2.imwrite('t.jpg', im_color)

tonemap1 = tone_mapping(E)
cv2.imwrite("b.jpg",tonemap1)

global_tonemap = cv2.pow(E/255.0, 100)
print (len(global_tonemap[global_tonemap==0].reshape(-1,1)))
cv2.imwrite("f.jpg",global_tonemap)'''
