import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import pandas as pd
from torch import int16

# data=np.genfromtxt('../data.csv',encoding='UTF-8',dtype='float32',delimiter=',')
# data[0][0]=25.0
#dataset numpy형태로 만들기
data=pd.read_csv('../data.csv',encoding='UTF-8')
data=data.to_numpy()
first=np.array([25,79])
data=np.vstack([first,data])
img_c = cv2.imread('../Image/house.bmp')
# def calc_dist(A,B):
#     assert len(A)==len(B)

#     sum=0
#     for a,b in zip(A,B):
#         sum+=(a-b)**2

#     return math.sqrt(sum)

# def calc_vector_mean(array):
#     summation=[0]*len(array[0])
#     for row in array:
#         for d in range(len(row)):
#             summation[d]+=row[d]
    
#     for s in range(len(summation)):
#         summation[s]=summation[s]/len(array)
    
#     return summation

# def calc_diff(A,B):
#     tol=0
#     for a,b in zip(A,B):
#         tol+=calc_dist(a,b)
#     return tol

# class Clustring(object):
#     def __init__(self,k,max_iter=10,tol=1e-4):
#         '''
#         k: 군집 수
#         max_iter:최대 스텝 횟수
#         tol:알고리즘을 계속 수행하기 위한 중심 업데이트 임계치
#         '''
#         self.k=k
#         self.max_iter=max_iter
#         self.tol=tol
    
#     def random_init(self,array):
#         M=[]
#         indices=[]
#         i=np.random.randint(0,len(array))

#         M.append(array[i])
#         indices.append(i)
#         while len(M)<self.k:
#             max_dist=-float('inf')
#             max_index=-1
#             for i in range(len(array)):
#                 avg_dist=0
#                 if i in indices:
#                     continue
#                 for j in range(len(M)):
#                     dist=calc_dist(array[i],array[j])
#                     avg_dist+=dist
#                 if max_dist<avg_dist:
#                     max_dist=avg_dist
#                     max_index=i

#                 # print(max_index)
#                 # print(max_dist)

#                 M.append(array[max_index])
#                 indices.append(max_index)
#         return M

#     def fit(self,X):
#         # X: Array of [10000,16]
#         # returns: cluster assignment of each vector, Centroids of cluster

#         self.centroids=self.random_init(X)

#         for iter in range(self.max_iter):
#             print(f'{iter+1} iteration...')
#             self._assign_cluster(X)
#             self._update_centroids(X)

#             if calc_diff(self.prev_centroids, self.centroids):
#                 break
        
#         return self.assignments, self.centroids
    
#     def _assign_cluster(self,X):
#         self.assignments=[]
#         for d in X:
#             min_dist=float('inf')
#             min_index=-1
#             for i,centroid in enumerate(self.centroids):
#                 dist=calc_dist(d,centroid)
#                 if dist<min_dist:
#                     min_dist=dist
#                     min_index=i
            
#             self.assignments.append(min_index)

#     def _update_centroids(self,X):
#         self.prev_centroids=np.copy(self.centroids)
#         for i in range(self.k):
#             data_indices=list(filter(lambda x: self.assignments[x]==i, range(len(self.assignments))))

#             if len(data_indices)==0:
#                 r=np.random.randint(0,len(X))
#                 self.centroids[i]=X[r]
#                 continue

#             cluster_data=[]
#             for index in data_indices:
#                 cluster_data.append(X[index])
#             self.centroids[i]=calc_vector_mean(cluster_data)


# c=Clustring(3)
# c.fit(data)
def getDistance(num1,num2):
    a = np.sum((num1-num2)**2)
    a=a**(1/2)
    return a

def KmeansClustering(img,k):
    value_num=img.shape[-1]
    cen=np.zeros([k,value_num])
    for i in range(k):
        rand_num=np.random.randint(0,255)
        for j in range(value_num):
            cen[i,j]=rand_num
    print(cen)
    count=0
    count_cluster=np.zeros(k)
    while True:

      new_cen=np.zeros(cen.shape)
      num_cen=np.zeros(k)
      label=np.zeros([img.shape[0],img.shape[1]])
      count+=1
      d=np.zeros(k)
      d=d.tolist()
      for y in range(img.shape[0]):
          for x in range(img.shape[1]):
            for j in range(k):
                d[j]=getDistance(img[y,x],cen[j])
            d=np.array(d)
            lbl=np.argmin(d)
            label[y,x]=lbl
            new_cen[lbl]+=img[y,x]
            num_cen[lbl]+=1
      new_cen+=1e-5
      num_cen+=1e-7  
      for j in range(k):
        new_cen[j]/=num_cen[j]
      if getDistance(cen[-1],new_cen[-1])<1:
          break     
      cen=np.copy(new_cen)
      print(new_cen)
      print(label)
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            for i in range(k):
                if (label[y,x]==i):
                    count_cluster[i]+=1
    print(count_cluster)
    return cen,label

def cluster_img(cen,label):
    img=np.zeros([label.shape[0],label.shape[1],3])
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            for j in range(cen.shape[0]):
                if(label[y,x]==j):
                    img[y,x]=cen[j]
    return img

    #   plt.subplot(4,2,count)
    #   plt.title('count_num: '+str(count))
    #   plt.plot(new_cen[:,0],new_cen[:,1],'ro')

# k=5
# min_num=25
# max_num=67
# value_num=3
# cen=np.random.randint(min_num,max_num,size=(k,value_num))
# print(cen)
# print(cen[2])
# print(getDistance(cen[1],cen[2]))
cen,label=KmeansClustering(img_c,5)
print(cen)
print(label)
b=cluster_img(cen,label)
b=b.astype(np.uint8)
b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
plt.imshow(b)
plt.show()

#img=np.random.randint(125,126,size=(10,10))
#plt.imshow(img)
#plt.show()