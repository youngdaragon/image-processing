import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img_c = cv2.imread('../Image/lena.bmp')

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

def getDistance2(in1,in2):
    return (sum((in1-in2)**2)**1/2)

def meanshift(data,k):
    random_num=np.random.rand(1)
    org_cen=np.array([(random_num*46+21),(random_num*74+5)]).reshape(-1)
    cen=np.copy(org_cen)
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
     weight_sum=np.zeros(2)
     count+=1
     d1=0
     for i in range(data.shape[0]):
        if (data[i,0]<cen[0]+k) and (cen[0]-k < data[i,0]):
            if (data[i,1]<cen[1]+k) and (cen[1]-k < data[i,1]):
              d1=getDistance2(data[i],cen)
              dis_sum+=d1
              weight_sum+=d1*data[i]
     new_cen=weight_sum/dis_sum
     if d1==0:
         print('아무점도 주변에 없습니다.')
         break
     if getDistance2(cen,new_cen)<1:
          break
     cen=np.copy(new_cen)
    print(cen,count)

meanshift(data,25)
    
    


        
    
    



# min 21, max 67      min 5, max 79