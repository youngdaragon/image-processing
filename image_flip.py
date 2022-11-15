import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/lena.jpg', cv2.IMREAD_GRAYSCALE) # input 이미지 입력 
# plt.title('lena')
# plt.imshow(img, cmap='gray')
# plt.show()
img2=np.zeros((img.shape[0],img.shape[1])) #반시계 90도 회전
img3=np.zeros((img.shape[0],img.shape[1])) #반시계 180도 회전
img4=np.zeros((img.shape[0],img.shape[1])) #반시계 270도 회전
a=input("숫자를 입력하세요: ")
image=np.zeros((img.shape[0],img.shape[1]))
if a=='90':
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img2[j,i]=img[i,j]
elif a=='180':
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img3[img.shape[0]-j-1,img.shape[1]-i-1]=img[j,i]

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img4[512-i-1,512-j-1]=img[j,i]
elif a=='270':
 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img4[i,img.shape[1]-j-1]=img[j,i]

try:
 if a=='90':
    image=img2
 elif a=='180':
    image=img3
 elif a=='270':
    image=img4
 elif a=='360' or a=='0':
    image=img
 else:
     raise Exception('0~360도 단위로만 입력해주세요')
 plt.title('lena')
 plt.imshow(image, cmap='gray')
 plt.show()
except Exception as e:                             # 예외가 발생했을 때 실행됨
    print('ERROR!',e)