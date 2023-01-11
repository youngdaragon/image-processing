import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1=cv2.imread('../Image/GT_img.jpg',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('../Image/GT_label.jpg',cv2.IMREAD_GRAYSCALE)
img3=cv2.imread('../Image/SVM_label.jpg',cv2.IMREAD_GRAYSCALE)
for i in range(img3.shape[0]):
    for j in range(img3.shape[1]):
        if (img3[i,j]==1):
            img3[i,j]=255
        elif (img3[i,j]==0):
            img3[i,j]=0

# count_0=0
# count_1=0
# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         if (img2[i,j]==1):
#             count_1+=1
#         elif (img2[i,j]==0):
#             count_0+=1

# print(count_0,count_1)
plt.imshow(img3,cmap='gray')
plt.show()