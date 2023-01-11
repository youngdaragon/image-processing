import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../wall/frame_0001.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../wall/frame_0002.png', cv2.IMREAD_GRAYSCALE)
sobel_y=np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
sobel_x=sobel_y.T
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

def mat_9(d):
    a=np.zeros([3,3])
    a=a+1
    b=np.zeros([d.shape[0],d.shape[1],9,2])
    for y in range(1,d.shape[0]-2):
        for x in range(1,d.shape[1]-2):
            b=1

img3=img1-img2
print(img3.shape)
d_x=Filtering(img1,sobel_x)
d_y=Filtering(img2,sobel_y)



