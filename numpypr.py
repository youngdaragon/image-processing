import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from pyparsing import java_style_comment


def filter(k, type):  # kernel making
    if type == 'gaussian':
        kernel = np.zeros((k, k))
        sigma = 1
        for i in range(k):
            for j in range(k):
                kx = j-math.floor(k)
                ky = i-math.floor(k)
                kernel[i, j] = math.exp(-(kx**2+ky**2)/(2*sigma**2))

    kernel /= np.sum(kernel)
    return kernel


def slide(kernel, img):
    h, w = img.shape
    h1, w1 = kernel.shape
    out_img = np.zeros(img.shape)
    k_size = math.floor(h1/2)
    for y in range(k_size, h-k_size):
        for x in range(k_size, w-k_size):
            sub_img = img[y-k_size:y+k_size+1, x-k_size:x+k_size+1]
            sub_img = kernel*sub_img
            out_img[y, x] = np.sum(sub_img)

    return out_img

def mosaica(img):
    h, w, c = img.shape
    newimg = np.zeros((h, w))
    for z in range(c):
        if(z == 0):
            for i in range(h):
                for j in range(w):
                    if(i % 2 == 0):
                        if(j % 2 == 0):
                            newimg[i, j] = img[i, j, z]
        elif(z == 1):
            for i in range(h):
                for j in range(w):
                    if(i % 2 == 0):
                        if(j % 2 != 0):
                            newimg[i, j] = img[i, j, z]
                    if(i % 2 != 0):
                        if(j % 2 == 0):
                            newimg[i, j] = img[i, j, z]
        else:
            for i in range(h):
                for j in range(w):
                    if(i % 2 != 0):
                        if(j % 2 != 0):
                            newimg[i, j] = img[i, j, z]
    return newimg

def mosaicb(img):
    h, w, c = img.shape
    newimg = np.zeros((h, w, c))
    for z in range(c):
        if(z == 0):
            for i in range(w):
                for j in range(h):
                    if(j%4==2):
                        if(i%2==0):
                            newimg[i,j,z]=img[i,j,z]
                    if(j%4==0):
                        if(i%2!=0):
                            newimg[i, j, z] = img[i, j, z]
        elif(z == 1):
            for i in range(w):
                for j in range(h):
                    if(j%2!=0):
                        newimg[i,j,z]=img[i,j,z]
        else:
            for i in range(w):
                for j in range(h):
                    if(j % 4 == 2):
                        if(i % 2 != 0):
                            newimg[i, j, z] = img[i, j, z]
                    if(j % 4 == 0):
                        if(i % 2 == 0):
                            newimg[i, j, z] = img[i, j, z]
    return newimg

def mosaic(img):
    h, w, c = img.shape
    newimg = np.zeros((h, w, c))
    for z in range(c):
        if z==0:
            for i in range(w):
                for j in range(h):
                    if(j%2==0):
                        if(i%4==1):
                            newimg[i,j,z]=img[i,j,z]
                    if(j%2!=0):
                        if(i%4==3):
                            newimg[i, j, z] = img[i, j, z]
        elif z==1:
            for i in range(w):
                for j in range(h):
                    if(j%2!=0):
                        if(i % 4 == 1 or i % 4 == 2):
                            newimg[i,j,z]=img[i,j,z]
                    if(j%2==0):
                        if(i % 4 == 0 or i % 4 == 3):
                            newimg[i, j, z] = img[i, j, z]
        else:
            for i in range(w):
                for j in range(h):
                    if(j % 2 == 0):
                        if(i % 4 == 2):
                            newimg[i, j, z] = img[i, j, z]
                    if(j % 2 != 0):
                        if(i % 4 == 0 ):
                            newimg[i, j, z] = img[i, j, z]
    return newimg
                

def mosaicd(img):
    h, w, c = img.shape
    newimg = np.zeros((h, w, c))
    for z in range(c):
        if(z == 0):
            for i in range(w):
                for j in range(h):
                    if(j % 3 == 2):
                            newimg[i, j, z] = img[i, j, z]
        elif(z == 1):
            for i in range(w):
                for j in range(h):
                    if(j % 3 == 1):
                        newimg[i, j, z] = img[i, j, z]
        else:
            for i in range(w):
                for j in range(h):
                    if(j % 3== 0):
                            newimg[i, j, z] = img[i, j, z]
    return newimg


def mosaice(img):
    h, w, c = img.shape
    newimg = np.zeros((h, w, c))
    for z in range(c):
        if(z == 0):
            for i in range(w):
                for j in range(h):
                    if(i%3==0):
                        if(j%3==0):
                            newimg[i, j, z] = img[i, j, z]
                    if(i%3==1):
                        if(j%3==1):
                            newimg[i, j, z] = img[i, j, z]
                    if(i % 3 == 2):
                        if(j % 3 == 2):
                            newimg[i, j, z] = img[i, j, z]
        elif(z == 1):
            for i in range(w):
                for j in range(h):
                    if(i % 3 == 0):
                        if(j % 3 == 2):
                            newimg[i, j, z] = img[i, j, z]
                    if(i % 3 == 1):
                        if(j % 3 == 0):
                            newimg[i, j, z] = img[i, j, z]
                    if(i % 3 == 2):
                        if(j % 3 == 1):
                            newimg[i, j, z] = img[i, j, z]
        else:
            for i in range(w):
                for j in range(h):
                    for j in range(h):
                        if(i % 3 == 0):
                            if(j % 3 == 1):
                                newimg[i, j, z] = img[i, j, z]
                        if(i % 3 == 1):
                            if(j % 3 == 2):
                                newimg[i, j, z] = img[i, j, z]
                        if(i % 3 == 2):
                            if(j % 3 == 0):
                                newimg[i, j, z] = img[i, j, z]
    return newimg
    

img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_COLOR)
newimg=mosaica(img)
plt.imshow(newimg,cmap='gray')
plt.show()

# def interpolation(img):
#     h,w,c=img.shape
#     newimg=np.zeros(img.shape)
#     for z in range(c):
#         kernel=filter(3,'gaussian')
#         newimg[:,:,z]=slide(kernel,img[:,:,z])
#         newimg=np.uint8(newimg)
#     return newimg
# new=interpolation(newimg)
# plt.imshow(new,cmap='gray',vmin=0,vmax=255)
# plt.show()