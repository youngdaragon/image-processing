import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_GRAYSCALE)
img=img.astype('float64')

def Gaussian_noise(img,sigma):
    a=sigma*np.random.randn(img.shape[0],img.shape[1])
    b=a+img
    return b

def Uniform_noise(img,sigma):
    a=sigma*2*(np.random.rand(img.shape[0],img.shape[1])-1)
    b=img+a
    return b


def MSE(img1,img2):
    MSE=np.sum((img1-img2)**2)/(img1.size)
    return MSE
def PSNR(img1,img2):
    M=MSE(img1,img2)
    PSNR=10*np.log10((255*255)/M)
    return PSNR

def Gaussian_filter(k_size,sigma):
 [kx,ky]=np.ogrid[-int(k_size/2):k_size-int(k_size/2),-int(k_size/2):k_size-int(k_size/2)]
 g_kernel=np.zeros((k_size,k_size))
 g_kernel=np.exp(-(kx**2+ky**2)/(2*sigma**2))
 g_kernel /=np.sum(g_kernel)
 return g_kernel

def Filtering(img,filter_x):
    out_image=np.zeros((img.shape[0],img.shape[1]))
    g_kernel=filter_x
    for y in range(int(filter_x.shape[0]/2),img.shape[0]-int(filter_x.shape[1]/2)):
        for x in range(int(filter_x.shape[0]/2),img.shape[1]-int(filter_x.shape[1]/2)):
            sub_image=img[y-int(filter_x.shape[0]/2):y+int(filter_x.shape[0]/2)+1,x-int(filter_x.shape[1]/2):x+int(filter_x.shape[1]/2)+1]
            sub_image=g_kernel*sub_image
            out_image[y,x]=np.sum(sub_image)
    return out_image

filter_x=Gaussian_filter(5, 4)
# b=Filtering(img, filter_x)
# print(b)
img1=img[3:8,3:8]
print(img1.dtype)
a=img[5,5]*np.ones([5,5],dtype=np.int32)
b=np.abs(img1-a)
b=np.exp(-((img1-a)**2)/(2*4**2))
c=b*filter_x
c/=np.sum(c)
print(c)
print(filter_x)
# plt.subplot(1,2,1)
# plt.imshow(a,cmap='gray')

# plt.subplot(1,2,2)
# plt.imshow(b,cmap='gray')
# plt.show()
