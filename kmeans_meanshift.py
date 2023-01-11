import numpy as np
import cv2 
import matplotlib.pyplot as plt

img_c = cv2.imread('../Image/lena.bmp')
img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
img_d = cv2.imread('../Image/house.bmp')
img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
img_1 = cv2.imread('../Image/FN1.bmp')
img_2 = cv2.imread('../Image/FN2.bmp')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
img_3 = cv2.imread('../Image/sh.png')

def getDistance(num1,num2):
    a = np.sum((num1-num2)**2)
    a=a**(1/2)
    return a

def KmeansClustering(img,k):
    value_num=img.shape[-1]
    cen=np.zeros([k,value_num])
    for i in range(k):
        rand_num_x=np.random.randint(0,img.shape[0])
        rand_num_y=np.random.randint(0,img.shape[1])
        cen[i]=img[rand_num_x,rand_num_y]
    print(cen)
    count=0
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
    print(num_cen)
    return cen,label

def KmeansClusteringYUV(img,k):
    value_num=img.shape[-1]
    cen=np.zeros([k,value_num])
    for i in range(k):
        rand_num_Y=np.random.randint(0,255)
        rand_num_U=np.random.randint(-111,111)
        rand_num_V=np.random.randint(-157,157)
        cen[i,0]=rand_num_Y
        cen[i,1]=rand_num_U
        cen[i,2]=rand_num_V
    print(cen)
    count=0
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
    print(num_cen)
    return cen,label

def KmeansClustering_XY(img,k):
    value_num=img.shape[-1]
    pixel_cen=np.zeros([k,value_num])
    cen=np.zeros([k,value_num+2])
    for i in range(k):
        rand_num=np.random.randint(0,255)
        rand_num_x=np.random.randint(0,img.shape[0])
        rand_num_y=np.random.randint(0,img.shape[1])
        rand_cord=np.array([rand_num_x,rand_num_y])
        for j in range(value_num):
            pixel_cen[i,j]=rand_num
        cen[i]=np.hstack((rand_cord,pixel_cen[i]))
    cen[0]=[305,945,162,22,0]
    cen[1]=[456,558,206,132,1]
    print(cen)
    count=0
    cordinate_img=np.zeros([img.shape[0],img.shape[1],5])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cordinate=np.array([y,x])
            cordinate_img[y,x]=np.hstack((cordinate,img[y,x]))
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
                d[j]=getDistance(cordinate_img[y,x],cen[j])
            d=np.array(d)
            lbl=np.argmin(d)
            label[y,x]=lbl
            new_cen[lbl]+=cordinate_img[y,x]
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
    print(num_cen)
    return cen,label

def cluster_imgXY(cen,label):
    img=np.zeros([label.shape[0],label.shape[1],3])
    new_cen=np.zeros([cen.shape[0],3])
    new_cen[:,0]=cen[:,2]
    new_cen[:,1]=cen[:,3]
    new_cen[:,2]=cen[:,4]
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            for j in range(cen.shape[0]):
                if(label[y,x]==j):
                    img[y,x]=new_cen[j]
    return img

def cluster_img(cen,label):
    img=np.zeros([label.shape[0],label.shape[1],3])
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            for j in range(cen.shape[0]):
                if(label[y,x]==j):
                    img[y,x]=cen[j]
    return img

def meanshift(img,k,win):
    value_num=img.shape[-1]
    cen=np.zeros([win,value_num])
    for i in range(win):
        rand_num=np.random.randint(0,255)
        for j in range(value_num):
            cen[i,j]=rand_num
    for i in range(win):
        rand_num_x=np.random.randint(0,img.shape[0])
        rand_num_y=np.random.randint(0,img.shape[1])
        cen[i]=img[rand_num_x,rand_num_y]
    print(cen)
    count=0
    count_label=np.zeros(win)
    while True:
     new_cen=np.zeros(cen.shape)
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
     if getDistance(cen[-1],new_cen[-1])<1.5:
          break
     cen=np.copy(new_cen)
     print(cen)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for n in range(win):
                if(label[i,j]==n):
                    count_label[n]+=1
    print(cen,count)
    print(count_label)
    return cen,label

def meanshiftYUV(img,k,win):
    value_num=img.shape[-1]
    cen=np.zeros([win,value_num])
    for i in range(win):
        rand_num_Y=np.random.randint(0,255)
        rand_num_U=np.random.randint(-111,111)
        rand_num_V=np.random.randint(-157,157)
        cen[i,0]=rand_num_Y
        cen[i,1]=rand_num_U
        cen[i,2]=rand_num_V
    print(cen)
    count=0
    count_label=np.zeros(win)
    while True:
     new_cen=np.zeros(cen.shape)
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
     if getDistance(cen[-1],new_cen[-1])<1.5:
          break
     cen=np.copy(new_cen)
     print(cen)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for n in range(win):
                if(label[i,j]==n):
                    count_label[n]+=1
    print(cen,count)
    print(count_label)
    return cen,label

def meanshift_XY(img,k,win):
    value_num=img.shape[-1]
    cen=np.zeros([win,value_num+2])
    cordinate_img=np.zeros([img.shape[0],img.shape[1],5])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cordinate=np.array([y,x])
            cordinate_img[y,x]=np.hstack((cordinate,img[y,x]))
    for i in range(win):
        rand_num_x=np.random.randint(0,img.shape[0])
        rand_num_y=np.random.randint(0,img.shape[1])
        cen[i]=cordinate_img[rand_num_x,rand_num_y]
    print(cen)
    count=0
    count_label=np.zeros(win)
    while True:
     new_cen=np.zeros(cen.shape)
     weight_sum=np.zeros(cen.shape)
     count+=1
     dis_sum=np.zeros(cen.shape[0])
     dis_sum=dis_sum.tolist()
     label=np.zeros([img.shape[0],img.shape[1]])
     for i in range(img.shape[0]):
         for j in range(img.shape[1]):
             for num in range(win):
                 if (cordinate_img[i,j,0]<cen[num,0]+k) and (cen[num,0]-k < cordinate_img[i,j,0]):
                     if (cordinate_img[i,j,1]<cen[num,1]+k) and (cen[num,1]-k < cordinate_img[i,j,1]):
                         if (cordinate_img[i,j,2]<cen[num,2]+k) and (cen[num,2]-k < cordinate_img[i,j,2]):
                             if (cordinate_img[i,j,3]<cen[num,3]+k) and (cen[num,3]-k < cordinate_img[i,j,3]):
                                 if (cordinate_img[i,j,4]<cen[num,4]+k) and (cen[num,4]-k < cordinate_img[i,j,4]):
                                     label[i,j]=num
                                     d=getDistance(cordinate_img[i,j],cen[num])
                                     dis_sum[num]+=d
                                     weight_sum[num]+=d*cordinate_img[i,j]

     dis_sum=np.array(dis_sum)
     weight_sum+=1e-5
     dis_sum+=1e-7
     for num in range(win):                        
         new_cen[num]=weight_sum[num]/dis_sum[num]
     if getDistance(cen[-1],new_cen[-1])<1.5:
          break
     cen=np.copy(new_cen)
     print(cen)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for n in range(win):
                if(label[i,j]==n):
                    count_label[n]+=1
    print(cen,count)
    print(count_label)
    return cen,label

def convert(img):
  img_r=img[:,:,0] 
  img_g=img[:,:,1]
  img_b=img[:,:,2]
  img_y=(0.299)*img_r+(0.587)*img_g+(0.114)*img_b
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          if(img_y[i,j]<0):
              img_y[i,j]=0
          elif(img_y[i,j]>255):
              img_y[i,j]=255
          else:
              img_y[i,j]=img_y[i,j]
  img_u=(-0.147)*img_r+(-0.289)*img_g+(0.436)*img_b
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          if(img_u[i,j]<-111):
              img_u[i,j]=-111
          elif(img_u[i,j]>111):
              img_u[i,j]=111
          else:
              img_u[i,j]=img_u[i,j]
  img_v=(0.615)*img_r+(-0.515)*img_g+(-0.100)*img_b
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          if(img_v[i,j]<-157):
              img_v[i,j]=-157
          elif(img_v[i,j]>157):
              img_v[i,j]=157
          else:
              img_v[i,j]=img_v[i,j]
  newimg=np.zeros(img.shape)
  newimg=newimg.astype(int)
  newimg[:,:,0]=img_y
  newimg[:,:,1]=img_u
  newimg[:,:,2]=img_v
  return newimg

# cen,label=KmeansClustering_XY(img_2,20)
# cen,label=meanshift_XY(img_1,80,40)
# b=cluster_imgXY(cen,label)
# b=b.astype(np.uint8)
# img_YUV=convert(img_c)
cen,label=KmeansClustering(img_3,3)
# cen,label=KmeansClusteringYUV(img_YUV,5)
# cen,label =meanshift(img_d,35,10)
# cen,label =meanshiftYUV(img_YUV,35,10)
b=cluster_img(cen,label)
b=b.astype(np.uint8)
b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
plt.imshow(b)
# plt.subplot(3,1,1)
# plt.imshow(b[:,:,0],cmap='gray')
# plt.subplot(3,1,2)
# plt.imshow(b[:,:,1],cmap='gray')
# plt.subplot(3,1,3)
# plt.imshow(b[:,:,2],cmap='gray')
plt.show()
