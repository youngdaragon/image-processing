import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.metrics import f1_score

img1=cv2.imread('../Image/GT_label.jpg',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('../Image/prewitt_label.jpg',cv2.IMREAD_GRAYSCALE)

tp=0
tn=0
fn=0
fp=0
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if (img1[i,j]==255):
            if(img2[i,j]==255):
                tp+=1
        if (img1[i,j]==255):
            if(img2[i,j]==0):
                fn+=1
        if (img1[i,j]==0):
            if(img2[i,j]==255):
                fp+=1
        if (img1[i,j]==0):
            if(img2[i,j]==0):
                tn+=1
            
print(tp,tn,fn,fp)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
print(precision)
print(recall)
f1=2*((precision*recall)/(precision+recall))
print(f1)