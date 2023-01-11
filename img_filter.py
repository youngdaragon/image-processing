import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_GRAYSCALE)

def Gaussian_noise(img,sigma):
    noise=sigma*np.random.randn(img.shape[0],img.shape[1])
    out_img=noise+img
    out_img[out_img>255]=255
    out_img[out_img<0]=0
    return out_img

def Impulse_noise(img,P):
    BWrad=0.5 #흑 백 나타나는 %게이지(black white  비율)
    bet=P/100 # 전체 이미지에서 몇퍼센트만큼 흑 백이 나올지
    noise=np.zeros(img.shape)
    num_black = np.round(bet * img.size * (1. - BWrad)) # 흑색(0)가 나타났을때 얼만큼의 픽셀에서 흑색 0이 나와야 하는지
    black_index = [np.random.randint(0, i - 1, int(num_black)) for i in img.shape] # 인덱스의 개수를 만들어 내기 위해 random함수를 사용하여 행,열 별로 인덱스를 랜덤으로 뽑는다.
    noise[black_index]=-255# 만들어낸 인덱스 값에 -255를 대입해 더했을때 0이나오게 한다.
    num_white=np.round(bet * img.size * BWrad) #백색(255)가 나타났을 때 얼만큼의 픽셀에서 백색 255가 나와야 하는지
    white_index = [np.random.randint(0, i - 1, int(num_white)) for i in img.shape] # 위 흑색과 동일하게 작업
    noise[white_index]=255 # 만들어진 인덱스 값에 +255를 넣어준다.
    out_img=noise+img
    out_img[out_img>255]=255
    out_img[out_img<0]=0
    return out_img

def MSE(img1,img2):
    MSE=np.sum((img1-img2)**2)/(img1.size)
    return MSE

def PSNR(img1,img2):
    M=MSE(img1,img2)
    PSNR=10*np.log10((255*255)/M)
    return PSNR

def Gaussian_filter(k_size,sigma):
 [kx,ky]=np.ogrid[-int(k_size/2):k_size-int(k_size/2),-int(k_size/2):k_size-int(k_size/2)]
 g_kernel=np.zeros((k_size,k_size))
 g_kernel=np.exp(-(kx**2+ky**2)/(2*sigma**2))
 g_kernel /=np.sum(g_kernel)
 return(g_kernel)

def Bilateral_filter(img,k_size,sigma_g,sigma_r):
    g_kernel=Gaussian_filter(k_size,sigma_g)
    out_image=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    r_kernel=np.zeros((k_size,k_size))
    for y in range(int(r_kernel.shape[0]/2),img.shape[0]-int(r_kernel.shape[1]/2)):
        for x in range(int(r_kernel.shape[0]/2),img.shape[1]-int(r_kernel.shape[1]/2)):
            sub_image=img[y-int(r_kernel.shape[0]/2):y+int(r_kernel.shape[0]/2)+1,x-int(r_kernel.shape[1]/2):x+int(r_kernel.shape[1]/2)+1]
            dif=sub_image-img[y,x]
            r_kernel=np.exp((-(dif**2))/(2*sigma_r**2))
            r_kernel/=np.sum(r_kernel)
            b_kernel=g_kernel*r_kernel
            b_kernel/=np.sum(b_kernel)
            sub_image1=b_kernel*sub_image
            out_image[y,x]=np.sum(sub_image1)
    for i in range(k_size//2):
      out_image[:,i]=img[:,i]
      out_image[i,:]=img[i,:]
      out_image[:,img.shape[1]-i-1]=img[:,img.shape[1]-i-1]
      out_image[img.shape[0]-i-1,:]=img[img.shape[0]-i-1,:]
    return out_image

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

noise_image1=Impulse_noise(img, 20)
noise_image1=np.float32(noise_image1)
noise_image2=Gaussian_noise(img, 50)
noise_image2=np.float32(noise_image2)
filter_g=Gaussian_filter(7,5)
gdenoise1=Filtering(noise_image1,filter_g)
gdenoise2=Filtering(noise_image2,filter_g)
bdenoise1=Bilateral_filter(noise_image1,7,5,120)
bdenoise2=Bilateral_filter(noise_image2,7,5,120)
opencv=cv2.bilateralFilter(noise_image1,7,5,120, borderType=cv2.BORDER_CONSTANT)
opencv2=cv2.bilateralFilter(noise_image2,7,5,120, borderType=cv2.BORDER_CONSTANT)
opencv3=cv2.GaussianBlur(noise_image1,ksize=(7,7),sigmaX=5, borderType=cv2.BORDER_CONSTANT)
opencv4=cv2.GaussianBlur(noise_image2,ksize=(7,7), sigmaX=5,borderType=cv2.BORDER_CONSTANT)
a= MSE(img,noise_image1)
b= PSNR(img,noise_image1)
print('impulse noise',' MSE: ',a,' PSNR: ',b)
c=MSE(img, noise_image2)
d=PSNR(img, noise_image2)
print('gausiann noise',' MSE: ',c,' PSNR: ',d)
e= MSE(img,gdenoise1)
f= PSNR(img,gdenoise1)
print('가우시안 필터를 이용해 임펄스 노이즈 제거','MSE: ',e,'PSNR: ',f)
g= MSE(img,gdenoise2)
h= PSNR(img,gdenoise2)
print('가우시안 필터를 이용해 가우시안 노이즈 제거','MSE: ',g,'PSNR: ',h)
i= MSE(img,bdenoise1)
j= PSNR(img,bdenoise1)
print('쌍방필터를 이용해 임펄스 노이즈 제거','MSE: ',i,'PSNR: ',j)
k= MSE(img,bdenoise2)
l= PSNR(img,bdenoise2)
print('쌍방 필터를 이용해 가우시안 노이즈 제거','MSE: ',k,'PSNR: ',l)
m= MSE(img,opencv)
n= PSNR(img,opencv)
print('opencv 쌍방 필터를 이용해 임펄스 노이즈 제거','MSE: ',e,'PSNR: ',f)
o= MSE(img,opencv2)
p= PSNR(img,opencv2)
print('opencv 쌍방 필터를 이용해 가우시안 노이즈 제거','MSE: ',g,'PSNR: ',h)
q= MSE(img,opencv3)
r= PSNR(img,opencv3)
print('opencv 가우시안 필터를 이용해 임펄스 노이즈 제거','MSE: ',i,'PSNR: ',j)
s= MSE(img,opencv4)
t= PSNR(img,opencv4)
print('opencv 가우시안 필터를 이용해 가우시안 노이즈 제거','MSE: ',k,'PSNR: ',l)
# plt.subplot(3,2,1)
# plt.title('lena')
# plt.hist(img.ravel(), 256, [0,256])
# plt.subplot(3,2,2)
# plt.title('gausian noise')
# plt.hist(noise_image2.ravel(), 256, [0,256])
# plt.subplot(3,2,3)
# plt.title('gausian filtering')
# plt.hist(gdenoise2.ravel(), 256, [0,256])
# plt.subplot(3,2,4)
# plt.title('bilateral filtering')
# plt.hist(bdenoise2.ravel(), 256, [0,256])
# plt.subplot(3,2,5)
# plt.title('opencv bilateral filtering')
# plt.hist(opencv2.ravel(), 256, [0,256])
# plt.subplot(3,2,6)
# plt.title('opencv gausian filtering')
# plt.hist(opencv4.ravel(), 256, [0,256])
plt.subplot(3,2,1)
plt.title('lena')
plt.imshow(img,cmap='gray')
plt.subplot(3,2,2)
plt.title('impulse noise lena')
plt.imshow(noise_image2,cmap='gray')
plt.subplot(3,2,3)
plt.title('bilateral filtering')
plt.imshow(bdenoise2,cmap='gray')
plt.subplot(3,2,4)
plt.title('gausian filtering')
plt.imshow(gdenoise2,cmap='gray')
plt.subplot(3,2,5)
plt.title('opencv gausian filtering')
plt.imshow(opencv4,cmap='gray')
plt.subplot(3,2,6)
plt.title('opencv bilateral filtering')
plt.imshow(opencv2,cmap='gray')
plt.show()