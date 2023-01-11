import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

img_1 = cv2.imread('../Image/5.jpg')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

def image_conv_function(img,filter):
    height, width= img.shape
    filtered_img=np.zeros((height,width))

    for i in range(2,height-2):
        for j in range(2,width-2):
            filtered_img[i,j]=np.sum(img[i-1:i+2,j-1:j+2]*filter)

    return filtered_img



prewitt_x=np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)
prewitt_y=prewitt_x.T
sobel_y=np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
sobel_x=sobel_y.T
x_d=image_conv_function(img_1,prewitt_x)
y_d=image_conv_function(img_1,prewitt_y)
f_d=(x_d**2+y_d**2)**(1/2)
for i in range(f_d.shape[0]):
    for j in range(f_d.shape[1]):
        if (f_d[i,j]<125):
            f_d[i,j]=0
        elif (f_d[i,j]>125):
            f_d[i,j]=255
a=image_conv_function(img_1,sobel_x)
b=image_conv_function(img_1,sobel_y)
c=(a**2+b**2)**(1/2)
for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        if (c[i,j]<125):
            c[i,j]=0
        elif (c[i,j]>125):
            c[i,j]=255
plt.imshow(c,cmap='gray')
cv2.imwrite('../Image/prewitt_label.jpg',f_d)
plt.show()