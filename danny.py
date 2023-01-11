import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

img = cv2.imread('../Image/5.jpg')
img2=img
def distance(init1, init2):
    dis = np.sum((init1-init2)**2)
    dis = dis**(1/2)
    return dis

def gaussian(x,y,h=1):
    return math.exp(-np.sum((x-y)**2)/h**2)

def meanshift(img,k,innit):
    w,h,c=img.shape
    data=img
    cen=innit
    cen=np.array(cen,dtype='uint8')
    newcen=cen
    t=newcen.shape[0]
    label=np.zeros((w,h))
    while True:
        dis=np.zeros((cen.shape[0]))
        weight=np.zeros((cen.shape))
        for i in range(w):
            for j in range(h):
                for z in range(t):
                    if distance(data[i,j],cen[z])<=k:
                        d=distance(data[i,j],cen[z])
                        dis[z]+=d
                        dis[z]+=0.0001
                        label[i,j]=z
                        weight[z]+=d*data[i,j]
        for q in range(newcen.shape[0]):
            newcen[q]=weight[q]//dis[q]
        if distance(cen[-1],newcen[-1])<5:
            break
        cen=np.copy(newcen)
        cen=np.array(cen,dtype=np.uint8)
        print(cen)
    return cen,label
def clustering(img,cen,label):
    w,h=label.shape
    k=cen.shape[0]
    data=img
    for x in range(w):
        for y in range(h):
            for q in range(k):
                if (label[x,y]==q):
                    data[x,y]=cen[q]
    return data
innit=np.random.randint(0,255,size=(30,3))
img2= cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
cen,label=meanshift(img,100,innit)
newimg=clustering(img,cen,label)
newimg= cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1)
plt.imshow(img2)
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(newimg)
plt.title('meanshift')
plt.show()