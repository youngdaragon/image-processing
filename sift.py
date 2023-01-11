import cv2
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
from sympy import laplace_transform
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_GRAYSCALE)

def downsampling(img):
    out_img=np.zeros(((img.shape[0])//2,(img.shape[1]//2)),dtype=np.int32)
    for y in range(out_img.shape[0]):
        for x in range(out_img.shape[1]):
            out_img[y,x]=(int(img[2*y,2*x])+int(img[2*y+1,2*x])+int(img[2*y,2*x+1])+int(img[2*y+1,2*x+1]))//4
    return out_img

def downsampling_k(img,K):
    out_img=downsampling(img)
    for i in range(K-1):
        out_img=downsampling(out_img)
    return out_img

def Gaussian_filter(k_size,sigma):
   [kx,ky]=np.ogrid[-int(k_size/2):k_size-int(k_size/2),-int(k_size/2):k_size-int(k_size/2)]
   g_kernel=np.zeros((k_size,k_size))
   g_kernel=np.exp(-(kx**2+ky**2)/(2*sigma**2))
   g_kernel /=np.sum(g_kernel)
   return(g_kernel)

def Filtering(img,filter_x):
    out_image=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    g_kernel=filter_x
    for y in range(int(filter_x.shape[0]/2),img.shape[0]-int(filter_x.shape[1]/2)):
        for x in range(int(filter_x.shape[0]/2),img.shape[1]-int(filter_x.shape[1]/2)):
            sub_image=img[y-int(filter_x.shape[0]/2):y+int(filter_x.shape[0]/2)+1,x-int(filter_x.shape[1]/2):x+int(filter_x.shape[1]/2)+1]
            sub_image1=g_kernel*sub_image
            out_image[y,x]=np.sum(sub_image1)
    for i in range(filter_x.shape[0]//2):
      out_image[:,i]=img[:,i]
      out_image[i,:]=img[i,:]
      out_image[:,img.shape[1]-i-1]=img[:,img.shape[1]-i-1]
      out_image[img.shape[0]-i-1,:]=img[img.shape[0]-i-1,:]
    return out_image


# laplacian=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])


# filter_g1=Gaussian_filter(5,1)
# filter_g2=Gaussian_filter(5,1.6)
# filter_g3=Gaussian_filter(5,1.6**2)
# filter_g4=Gaussian_filter(5,1.6**3)
# filter_g5=Gaussian_filter(5,1.6**4)
# filter_g6=Gaussian_filter(5,1.6**5)

# g_img1=Filtering(img,filter_g1)
# g_img2=Filtering(img,filter_g2)
# g_img3=Filtering(img,filter_g3)
# g_img4=Filtering(img,filter_g4)
# g_img5=Filtering(img,filter_g5)
# g_img6=Filtering(img,filter_g6)
# plt.subplot(1,2,1)
# plt.imshow(g_img5,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(g_img6,cmap='gray')
# plt.subplot(3,2,3)
# plt.imshow(g_img3,cmap='gray')
# plt.subplot(3,2,4)
# plt.imshow(g_img4,cmap='gray')
# plt.subplot(3,2,5)
# plt.imshow(g_img5,cmap='gray')
# plt.subplot(3,2,6)
# plt.imshow(g_img6,cmap='gray')
# dog1=g_img1-g_img2
# dog2=g_img2-g_img3
# dog3=g_img3-g_img4
# dog4=g_img4-g_img5
# dog5=g_img5-g_img6
# log_img1=Filtering(g_img1,laplacian)
# log_img2=Filtering(g_img2,laplacian)
# log_img3=Filtering(g_img3,laplacian)
# log_img4=Filtering(g_img4,laplacian)
# log_img5=Filtering(g_img5,laplacian)
# log_img6=Filtering(g_img6,laplacian)
# plt.subplot(3,2,1)
# plt.imshow(log_img2-dog1,cmap='gray')
# plt.subplot(3,2,2)
# plt.imshow(log_img3-dog2,cmap='gray')
# plt.subplot(3,2,3)
# plt.imshow(log_img4-dog3,cmap='gray')
# plt.subplot(3,2,4)
# plt.imshow(log_img5-dog4,cmap='gray')
# plt.subplot(3,2,5)
# plt.imshow(log_img6-dog5,cmap='gray')
# plt.subplot(3,2,6)
# plt.imshow(log_img6-dog5,cmap='gray')
# plt.subplot(1,2,1)
# plt.imshow(dog5,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(dog4,cmap='gray')
# plt.subplot(3,2,3)
# plt.imshow(dog3,cmap='gray')
# plt.subplot(3,2,4)
# plt.imshow(dog4,cmap='gray')
# plt.subplot(3,2,5)
# plt.imshow(dog5,cmap='gray')
# plt.subplot(3,2,6)
# plt.imshow(dog5,cmap='gray')
# plt.subplot(1,2,1)
# plt.imshow(log_img5,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(log_img6,cmap='gray')
# plt.subplot(3,2,3)
# plt.imshow(log_img3,cmap='gray')
# plt.subplot(3,2,4)
# plt.imshow(log_img4,cmap='gray')
# plt.subplot(3,2,5)
# plt.imshow(log_img5,cmap='gray')
# plt.subplot(3,2,6)
# plt.imshow(log_img6,cmap='gray')

sampling_img1=downsampling_k(img,1)
sampling_img2=downsampling_k(img,2)

# plt.title('octave 0')
# plt.imshow(img,cmap='gray')
# plt.title('octave 1')
# plt.imshow(sampling_img1,cmap='gray')
# plt.title('octave 2')
# plt.imshow(sampling_img2,cmap='gray')
# plt.show()

def sift(img):
    sampling_img1=downsampling_k(img,1)
    sampling_img2=downsampling_k(img,2)
    
    octave1=np.zeros((img.shape[0],img.shape[1],6))
    octave2=np.zeros((sampling_img1.shape[0],sampling_img1.shape[1],6))
    octave3=np.zeros((sampling_img2.shape[0],sampling_img2.shape[1],6))
    
    for i in range(6):
        octave1[:,:,i]=Filtering(img,Gaussian_filter(5,1.6**i))
        octave2[:,:,i]=Filtering(sampling_img1,Gaussian_filter(5,1.6**i))
        octave3[:,:,i]=Filtering(sampling_img2,Gaussian_filter(5,1.6**i))
    
    DoG_octave1=np.zeros((img.shape[0],img.shape[1],5))
    DoG_octave2=np.zeros((sampling_img1.shape[0],sampling_img1.shape[1],5))
    DoG_octave3=np.zeros((sampling_img2.shape[0],sampling_img2.shape[1],5))
    
    for i in range(5):
        DoG_octave1[:,:,i]=octave1[:,:,i]-octave1[:,:,i+1]
        DoG_octave2[:,:,i]=octave2[:,:,i]-octave2[:,:,i+1]
        DoG_octave3[:,:,i]=octave3[:,:,i]-octave3[:,:,i+1]
    point_ls1=[]
    point_ls2=[]
    point_ls3=[]
    for i in range(1,4):
        for y in range(1,DoG_octave1.shape[0]-2):
            for x in range(1,DoG_octave1.shape[1]-1):
                point_f=[DoG_octave1[y-1:y+2,x-1:x+2,i-1],DoG_octave1[y-1:y+2,x-1:x+2,i],DoG_octave1[y-1:y+2,x-1:x+2,i+1]]
                max=np.max(point_f)
                min=np.min(point_f)
                if(max==DoG_octave1[y,x,i] or min==DoG_octave1[y,x,i]):
                    if(x!=1 and y!=1 and x!=DoG_octave1.shape[1]-2 and y!=DoG_octave1.shape[0]-2):
                        point_ls1.append([y,x,i,1])
    for i in range(1,4):
        for y in range(1,DoG_octave2.shape[0]-2):
            for x in range(1,DoG_octave2.shape[1]-2):
                point_f=[DoG_octave2[y-1:y+2,x-1:x+2,i-1],DoG_octave2[y-1:y+2,x-1:x+2,i],DoG_octave2[y-1:y+2,x-1:x+2,i+1]]
                max=np.max(point_f)
                min=np.min(point_f)
                if(max==DoG_octave2[y,x,i] or min==DoG_octave2[y,x,i]):
                    if(x!=1 and y!=1 and x!=DoG_octave2.shape[1]-2 and y!=DoG_octave2.shape[0]-2):
                        point_ls2.append([2*y,2*x,i,2])
    for i in range(1,4):
        for y in range(1,DoG_octave3.shape[0]-2):
            for x in range(1,DoG_octave3.shape[1]-2):
                point_f=[DoG_octave3[y-1:y+2,x-1:x+2,i-1],DoG_octave3[y-1:y+2,x-1:x+2,i],DoG_octave3[y-1:y+2,x-1:x+2,i+1]]
                max=np.max(point_f)
                min=np.min(point_f)
                if(max==DoG_octave3[y,x,i] or min==DoG_octave3[y,x,i]):
                    if(x!=1 and y!=1 and x!=DoG_octave3.shape[1]-2 and y!=DoG_octave3.shape[0]-2):
                        point_ls3.append([4*y,4*x,i,3])
    point_feature=point_ls1+point_ls2+point_ls3
    point_feature=np.array(point_feature)
    return point_feature
a=sift(sampling_img1)
print(a.shape)
print(a.size)
print(a[:,0])
print(a[:,1])
print(a)
plt.figure(1)
plt.imshow(img,cmap="gray")
plt.plot(a[:,1],a[:,0],'ro')
plt.show()

    





