import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data=np.genfromtxt('../data.csv',encoding='UTF-8',dtype='float32',delimiter=',')
# data[0][0]=25.0
#dataset numpy형태로 만들기
data=pd.read_csv('../data.csv',encoding='UTF-8')
data=data.to_numpy()
first=np.array([25,79])
data=np.vstack([first,data])
img_c = cv2.imread('../Image/lena.bmp')

def getDistance2(in1,in2):
    return sum((in1-in2)**2)

def kmeans(data):
    cen=np.array([[20,20],[30,30],[40,40]])
    count=0
    while True:

      new_cen=np.zeros(cen.shape)
      num_cen=np.zeros(3)
      label=np.zeros(data.shape[0])
      count+=1
      for i in range(data.shape[0]):
          d1=getDistance2(data[i],cen[0,:])
          d2=getDistance2(data[i],cen[1,:])
          d3=getDistance2(data[i],cen[2,:])

          if d1<d2 and d1<d3:
              lbl=0
          elif d2<d1 and d2<d3:
              lbl=1
          else:
              lbl=2
          label[i]=lbl
          new_cen[lbl,:]+=data[i,:]
          num_cen[lbl]+=1
      new_cen[0,:]/=num_cen[0]
      new_cen[1,:]/=num_cen[1]
      new_cen[2,:]/=num_cen[2]
      if getDistance2(cen[-1],new_cen[-1])<1:
          break     
      cen=np.copy(new_cen)
      print(new_cen)
      print(label)
      plt.subplot(4,2,count)
      plt.title('count_num: '+str(count))
      plt.plot(new_cen[:,0],new_cen[:,1],'ro')
    
cen=np.array([[20,20],[30,30],[40,40]])
print(cen[-1])
# print(img_c.shape[-1])
# kmeans(data)
# plt.show()















         






