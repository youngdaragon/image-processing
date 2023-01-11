import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/chess.png', cv2.IMREAD_GRAYSCALE)

def image_conv_function(img,filter):
    height, width= img.shape
    filtered_img=np.zeros((height,width))

    for i in range(2,height-2):
        for j in range(2,width-2):
            filtered_img[i,j]=np.sum(img[i-1:i+2,j-1:j+2]*filter)

    return filtered_img
#y로 미분한 영상을 제곱
#y로 미분한 영상과 x로 미분한 영상을 곱함

def meshigrid_generate_Gaussian_filter(k,sigma):
    k=k//2
    x=np.arange(-k,k+1)
    y=np.arange(-k,k+1)
    [kx,ky]=np.meshgrid(x,y)
    filter=np.exp(-(kx**2+ky**2)/(2*sigma**2))
    filter /=np.sum(filter)

    return filter

def harris_corner_detector(gray_img):
    height,width=gray_img.shape
    #미분 kernel
    hor_dif=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ver_dif=hor_dif.transpose()
    K=0.04
    # 가우시안 필터 생성
    gaussian=meshigrid_generate_Gaussian_filter(3,0.5)
    #1차 미분
    x_diff=image_conv_function(gray_img,hor_dif)
    y_diff=image_conv_function(gray_img,ver_dif)
    #2차 미분
    x_2_diff=x_diff**2
    y_2_diff=y_diff**2
    xy_diff=x_diff*y_diff
    #가우시안 필터 취하기
    gau_x_2_diff=image_conv_function(x_2_diff,gaussian)
    gau_y_2_diff=image_conv_function(y_2_diff,gaussian)
    gau_xy_diff=image_conv_function(xy_diff,gaussian)
    #det,trace 계산후 C값 정하고 normalize 하기
    det=(gau_x_2_diff*gau_y_2_diff)-(gau_xy_diff**2)
    trace_squa=(gau_x_2_diff + gau_y_2_diff)**2
    C=det-(K*trace_squa)
    x=np.where(C>0.1)

    return x

corner_map=harris_corner_detector(img)
print(corner_map)
# plt.figure(1)
# plt.imshow(img,cmap="gray")
# plt.plot(corner_map[1],corner_map[0],'ro')
# plt.show()


