import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd
from sklearn import metrics

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
lr=0.00001
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

