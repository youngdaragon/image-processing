import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img_c = cv2.imread('../Image/house.bmp')
# data=np.genfromtxt('../data.csv',encoding='UTF-8',dtype='float32',delimiter=',')
# data[0][0]=25.0
# dataset numpy형태로 만들기
data=pd.read_csv('../data.csv',encoding='UTF-8')
data=data.to_numpy()
first=np.array([25,79])
data=np.vstack([first,data])

# print(data[:,0])
# random_num=np.random.rand(1)
# org_cen=np.array([(random_num*46+21),(random_num*74+5)]).reshape(-1)
# print(org_cen)
# print(org_cen[1])
# 가중치를 이용해서 중심점으로 이동 sum(거리*y)/sum(거리) 해주면 됨 

def getDistance(num1,num2):
    a = np.sum((num1-num2)**2)
    a=a**(1/2)
    return a

def meanshift(img,k,win):
    value_num=img.shape[-1]
    cen=np.zeros([win,value_num])
    for i in range(win):
        rand_num=np.random.randint(0,255)
        for j in range(value_num):
            cen[i,j]=rand_num
    print(cen)
    count=0
    while True:
    # 가우시안 커널 사용시
    # k=k//2
    # x=np.arange(-k,k+1)
    # y=np.arange(-k,k+1)
    # sigma=5
    # [kx,ky]=np.meshgrid(x,y)
    # kernel=np.exp(-(kx**2+ky**2)/(2*sigma**2))
    # kernel /=np.sum(kernel)
     new_cen=np.zeros(cen.shape)
     dis_sum=0
     weight_sum=np.zeros(cen.shape)
     count+=1
     dis_sum=np.zeros(cen.shape[0])
     dis_sum=dis_sum.tolist()
     label=np.zeros([img.shape[0],img.shape[1]])
     for i in range(img.shape[0]):
         for j in range(img.shape[1]):
             for num in range(win):
                 if (img[i,j,0]<cen[num,0]+k) and (cen[num,0]-k < img[i,j,0]):
                     if (img[i,j,1]<cen[num,1]+k) and (cen[num,1]-k < img[i,j,1]):
                         if (img[i,j,2]<cen[num,2]+k) and (cen[num,2]-k < img[i,j,2]):
                             label[i,j]=num
                             d=getDistance(img[i,j],cen[num])
                             dis_sum[num]+=d
                             weight_sum[num]+=d*img[i,j]

     dis_sum=np.array(dis_sum)
     weight_sum+=1e-5
     dis_sum+=1e-7
     for num in range(win):                        
         new_cen[num]=weight_sum[num]/dis_sum[num]
     if getDistance(cen[-1],new_cen[-1])<1.8:
          break
     cen=np.copy(new_cen)
     print(cen)
    print(cen,count)
    return cen,label

cen,label =meanshift(img_c,50,5)

def cluster_img(cen,label):
    img=np.zeros([label.shape[0],label.shape[1],3])
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            for j in range(cen.shape[0]):
                if(label[y,x]==j):
                    img[y,x]=cen[j]
    return img

b=cluster_img(cen,label)
b=b.astype(np.uint8)
b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
plt.imshow(b)
plt.show()