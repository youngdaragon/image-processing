import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd
from sklearn import metrics

img_1 = cv2.imread('../Image/4.jpg')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
label=cv2.imread('../Image/GT_label.jpg',cv2.IMREAD_GRAYSCALE)
slp_kernel=cv2.imread('../Image/slp.jpg',cv2.IMREAD_GRAYSCALE)

def f1(img1,img2):
    tp=0
    tn=0
    fn=0
    fp=0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (img1[i,j]==1):
                if(img2[i,j]==1):
                    tp+=1
            if (img1[i,j]==1):
                if(img2[i,j]==0):
                    fn+=1
            if (img1[i,j]==0):
                if(img2[i,j]==1):
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
    return f1

def cannyedge(img):
    edge_map = cv2.Canny(img, 100, 150)
    label=np.zeros(edge_map.shape)
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if (edge_map[i,j]<125):
                label[i,j]=0
            elif (edge_map[i,j]>125):
                label[i,j]=1
    cv2.imwrite('../Image/GT_img.jpg',edge_map)
    cv2.imwrite('../Image/GT_label.jpg',label)
    return edge_map

def image_conv_function(img,filter):
    height, width= img.shape
    filtered_img=np.zeros((height,width))

    for i in range(2,height-2):
        for j in range(2,width-2):
            filtered_img[i,j]=np.sum(img[i-1:i+2,j-1:j+2]*filter)

    return filtered_img


def prewitt(img):
    prewitt_x=np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)
    prewitt_y=prewitt_x.T
    x_d=image_conv_function(img,prewitt_x)
    y_d=image_conv_function(img,prewitt_y)
    f_d=(x_d**2+y_d**2)**(1/2)
    for i in range(f_d.shape[0]):
        for j in range(f_d.shape[1]):
            if (f_d[i,j]<125):
                f_d[i,j]=0
            elif (f_d[i,j]>125):
                f_d[i,j]=255
    cv2.imwrite('../Image/prewitt_label.jpg',f_d)
    return f_d

def sobel(img):
    sobel_y=np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
    sobel_x=sobel_y.T
    a=image_conv_function(img,sobel_x)
    b=image_conv_function(img,sobel_y)
    c=(a**2+b**2)**(1/2)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if (c[i,j]<125):
                c[i,j]=0
            elif (c[i,j]>125):
                c[i,j]=255
    cv2.imwrite('../Image/prewitt_label.jpg',c)
    return c

def svm_Kernel(data,label):
    x=data.reshape(-1,3)
    y=label.reshape(-1)

    # clf=svm.SVC(kernel='linear',C=1)
    clf2=svm.SVC(kernel='rbf', gamma=0.7, C=1)
    # clf3=svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)
    # clf4=svm.LinearSVC(C=1, max_iter=10000)
    clf2.fit(x,y)

    Yte_pred_linear_c1 = clf2.predict(x)

    Yte_pred_linear_c1=Yte_pred_linear_c1.reshape(data.shape[0],data.shape[1])
    cv2.imwrite(Yte_pred_linear_c1)

    return Yte_pred_linear_c1


for i in range(slp_kernel.shape[0]):
        for j in range(slp_kernel.shape[1]):
            if (slp_kernel[i,j]==0):
                slp_kernel[i,j]=0
            elif (slp_kernel[i,j]==255):
                slp_kernel[i,j]=1

a=f1(slp_kernel,label)
print(a)

data = cv2.imread('../Image/4.jpg')
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
label=cv2.imread('../Image/GT_label.jpg',cv2.IMREAD_GRAYSCALE)
cordinate_img=np.zeros([data.shape[0],data.shape[1],5])
for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            cordinate=np.array([y,x])
            cordinate_img[y,x]=np.hstack((cordinate,data[y,x]))
data=cordinate_img.reshape(-1,5)
label=label.reshape(-1)


class slp:
    def __init__(self,size):
        self.w=np.random.rand(size,1)
        self.b=np.random.rand(1)

    def forward(self,input):
        pred=input@self.w+self.b
        y=1 if pred>0 else 0
        return y

    def backward(self,label,x,lr):
        for i in range(len(x)):
            if label==0:
                label=-1

            self.w[i]+=lr*label*x[i]
            self.b+=lr*label

    def getw(self):
        return self.w

    def getb(self):
        return self.b
lr=0.001
max_epochs=50
model=slp(5)
acc=np.zeros([max_epochs])
for epochs in range(max_epochs):
    np.random.shuffle(data)
    correct=0
    for i in range(len(label)):
        pred=model.forward(data[i,:])

        if pred == label[i]:
            correct+=1
        
        else:
            model.backward(label[i],data[i,:],lr)
    acc[epochs]=correct/len(label)
    print("Epochs : {}, acc : {}".format(epochs,acc[epochs]))

print(model.getw())
out_img=np.zeros(label.shape)
out_img=data@model.getw()+model.getb()
out_img=out_img.reshape(224,224)
out_img[out_img<=0]=0
out_img[out_img>0]=255
cv2.imwrite('../Image/slp.jpg',out_img)
plt.imshow(out_img,cmap='gray')
plt.show()
