import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/lena.bmp',cv2.IMREAD_GRAYSCALE)
#1
# img=np.float32(img)
# h,w=img.shape
# s=2
# sh=h*s
# sw=w*s

# image_s=np.zeros([sh,sw])
# for sy in range(sh):
#     for sx in range(sw):
#         #neareset neighborhood
#         y=sy//2
#         x=sx//2
#         if(sy%2==0 and sx%2==0) or (sy==sh-1) or (sx==sw-1):   
#          image_s[sy,sx]=img[y,x]
#         elif(sy%2==1 and sx%2==0):
#             image_s[sy,sx]=(img[y,x]+img[y+1,x])//2
#         elif(sy%2==0 and sx%2==1):
#             image_s[sy,sx]=(img[y,x]+img[y,x+1])//2
#         elif(sy%2==1 and sx%2==1):
#              image_s[sy,sx]=(img[y,x]+img[y+1,x]+img[y,x+1]+img[y+1,x+1])//4

        

# plt.figure(1)
# plt.imshow(img,cmap='gray',vmin=0,vmax=255)
# plt.figure(2)
# plt.imshow(image_s,cmap='gray',vmin=0,vmax=255)
# plt.show()
#2

def Filtering(img,filter_x):
    out_image=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    # out_image=np.copy(img)
    g_kernel=filter_x
    for y in range(int(filter_x.shape[0]/2),img.shape[0]-int(filter_x.shape[1]/2)):
        for x in range(int(filter_x.shape[0]/2),img.shape[1]-int(filter_x.shape[1]/2)):
            sub_image=img[y-int(filter_x.shape[0]/2):y+int(filter_x.shape[0]/2)+1,x-int(filter_x.shape[1]/2):x+int(filter_x.shape[1]/2)+1]
            sub_image1=g_kernel*sub_image
            out_image[y,x]=np.sum(sub_image1)
    # out_image[out_image>255]=255
    # out_image[out_image<0]=0
    return out_image
prewitt_x=np.array([-1,0,1,1,0,1,-1,0,1]).reshape(3,3)
prewitt_y=prewitt_x.T

sobel_y=np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
sobel_x=sobel_y.T

a=Filtering(img, prewitt_x)
b=Filtering(img, prewitt_y)
c=Filtering(img, sobel_x)
d=Filtering(img, sobel_y)
plt.subplot(2,2,1)
plt.imshow(a,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(b,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(c,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(d,cmap='gray')
plt.show()

