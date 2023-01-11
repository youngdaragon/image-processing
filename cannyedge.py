import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img_1 = cv2.imread('../Image/5.jpg')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

edge_map = cv2.Canny(img_1, 100, 150)
label=np.zeros(edge_map.shape)
for i in range(edge_map.shape[0]):
    for j in range(edge_map.shape[1]):
        if (edge_map[i,j]<125):
            label[i,j]=0
        elif (edge_map[i,j]>125):
            label[i,j]=255
cv2.imwrite('../Image/GT_img.jpg',edge_map)
cv2.imwrite('../Image/GT_label.jpg',label)
plt.imshow(edge_map,cmap='gray')
plt.show()