import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math

img_c = cv2.imread('../Image/lena.bmp')
fix_img = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

def convert(img,a,b,c):
  img_r=img[:,:,0] 
  img_g=img[:,:,1]
  img_b=img[:,:,2]
  img_y=a*img_r+b*img_g+c*img_b
  return img_y
img=convert(img_c,0.3,0.5,0.2)
# img2=convert(img,0.3,0.5,0.2)
# plt.title('lena')
# plt.imshow(img2,cmap='gray')
# plt.show()
# def offset(img,offset):
#     img_gray=convert(img,0.3,0.5,0.2)
#     offset_img=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
#     for x in range(img.shape[0]):
#         for y in range(img.shape[1]):
#             offset_img[x,y]=img_gray[x,y]-offset
#             if offset_img[x,y] <= 0:
#                 offset_img[x,y]=0
#     return offset_img
def offset(img,offset):
    img=np.float64(img)
    out=img+offset
    out[out>255]=255
    out[out<0]=0
    return out
# img3=offset(img,500)
# img4=offset(img,-500)
# plt.title('lena')
# plt.subplot(1,2,1)
# plt.imshow(img3,cmap='gray')
# plt.title('lena')
# plt.subplot(1,2,2)
# plt.imshow(img4,cmap='gray')
# plt.show()

# def hist(img,N):
#     img_gray=convert(img,0.3,0.5,0.2)
#     size=img.shape[0]*img.shape[1]
#     hist=np.zeros(shape=(size,),dtype=np.uint8)
#     img_1d=img_gray.reshape(-1)
#     histogram=plt.hist(img_1d, bins=N, label='lena')
#     return histogram
def hist(img,N):
    hist=np.zeros(N)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n=int(img[i][j])
            hist[n]=hist[n]+1
    return hist
# histogram=hist(img,255)
# print(histogram)
# plt.bar(histogram,300)
# plt.show()


def mosaicked_img(color_img):
    converted_img=np.zeros((512,512),dtype=np.uint8)
    R=color_img[:,:,0]
    G=color_img[:,:,1]
    B=color_img[:,:,2]

    for i in range(256):
        for j in range(256):
            converted_img[2*i][2*j+1]=R[2*i][2*j+1]
            converted_img[2*i+1][2*j]=B[2*i+1][2*j]
            converted_img[2*i+1][2*j+1]=G[2*i+1][2*j+1]
            converted_img[2*i][2*j]=G[2*i][2*j]
    return converted_img


converted_img=mosaicked_img(img_c)
plt.subplot(1,2,1)
plt.imshow(converted_img)


image_new=np.zeros((512,512),dtype=np.uint8)
image_new[1::2,1::2]=img_c[1::2,1::2,0]
image_new[1::2,0::2]=img_c[1::2,0::2,1]
image_new[0::2,1::2]=img_c[0::2,1::2,1]
image_new[0::2,0::2]=img_c[0::2,0::2,2]
plt.subplot(1,2,2)
plt.imshow(image_new)
plt.show()
# def mosaic(img):
#     h,w,c=img.shape
#     newimg=np.zeros((h,w,c))
#     for z in range(c):
#         if(z==0):
#             for i in range(h):
#                 for j in range(w):
#                     if(i%2==0):
#                         if(j%2==0):
#                             newimg[i,j,z]=img[i,j,z]
#         elif(z==1):
#             for i in range(h):
#                 for j in range(w):
#                     if(i % 2 == 0):
#                         if(j % 2 != 0):
#                             newimg[i,j,z]=img[i,j,z]
#                     if(i%2!=0):
#                         if(j%2==0):
#                             newimg[i, j,z] = img[i, j, z]
#         else:
#             for i in range(h):
#                 for j in range(w):
#                     if(i % 2 != 0):
#                         if(j % 2 != 0):
#                             newimg[i, j,z] = img[i, j, z]
#     return newimg
# newimg=mosaic(img)
# print(newimg.shape)
# plt.subplot(1,4,1)
# plt.imshow(newimg[:,: ,0],cmap='gray')
# plt.title('red')
# plt.subplot(1,4,2)
# plt.imshow(newimg[:, :, 1],cmap='gray')
# plt.title('green')
# plt.subplot(1,4,3)
# plt.imshow(newimg[:, :, 2],cmap='gray')
# plt.title('blue')
# plt.subplot(1,4,4)
# plt.imshow(newimg)
# plt.show()
