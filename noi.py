import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_GRAYSCALE)
# P=30

# BWrad=0.5 #흑 백 나타나는 %게이지(black white  비율)
# bet=P/100 # 전체 이미지에서 몇퍼센트만큼 흑 백이 나올지
# out_img=np.copy(img)
# num_black = np.round(bet * img.size * (1. - BWrad)) # 흑색(0)가 나타났을때 얼만큼의 픽셀에서 흑색 0이 나와야 하는지
# black_index = [np.random.randint(0, i - 1, int(num_black)) for i in img.shape] # 인덱스의 개수를 만들어 내기 위해 random함수를 사용하여 행,열 별로 인덱스를 랜덤으로 뽑는다.
# out_img[black_index] = 0 # 만들어낸 인덱스 값에 0을 대입해 블랙인 점을 만듦
# num_white=np.round(bet * img.size * BWrad) #백색(255)가 나타났을 때 얼만큼의 픽셀에서 백색 255가 나와야 하는지
# white_index = [np.random.randint(0, i - 1, int(num_white)) for i in img.shape] # 위 흑색과 동일하게 작업
# out_img[white_index] = 255 #위 흑색과 동일하게 작업하여 255인 백색점을 뽑아냄


# # plt.title('sb')
# # plt.imshow(out_img,cmap='gray')
# # plt.show()
# # count=0
# # print(img.shape)
# # for i in img.shape:
# #     count+=1
# #     print(i)
# #     a=np.random.randint(0,i-1,int(num_black))
# #     print(a)
# # print(count)
# # a=np.zeros((5,5))
# # b=[[1,2,3,4],[1,2,3,4]]
# # a[b]=1
# # print(a)
# print(np.count_nonzero(out_img==255))
# print(np.count_nonzero(out_img==0))
# # a=np.random.randint(0,5,10)
# # print(a)

# def Bilateral_filter(img,k_size,sigma_g,sigma_r):
#     [kx,ky]=np.ogrid[-int(k_size/2):k_size-int(k_size/2),-int(k_size/2):k_size-int(k_size/2)]
#     g_kernel=np.zeros((k_size,k_size))
#     g_kernel=np.exp(-(kx**2+ky**2)/(2*sigma_g**2))
#     out_image=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
#     r_kernel=np.zeros((k_size,k_size))
#     for y in range(int(r_kernel.shape[0]/2),img.shape[0]-int(r_kernel.shape[1]/2)):
#         for x in range(int(r_kernel.shape[0]/2),img.shape[1]-int(r_kernel.shape[1]/2)):
#             sub_image=img[y-int(r_kernel.shape[0]/2):y+int(r_kernel.shape[0]/2)+1,x-int(r_kernel.shape[1]/2):x+int(r_kernel.shape[1]/2)+1]
#             dif=sub_image-img[y,x]
#             r_kernel=np.exp(-(dif*dif)/2*sigma_r*sigma_r)
#             b_kernel=g_kernel*r_kernel
#             b_kernel/=np.sum(b_kernel)
#             sub_image1=b_kernel*sub_image
#             out_image[y,x]=np.sum(sub_image1)
#     return out_image
# def Gaussian_filter(k_size,sigma):
#  [kx,ky]=np.ogrid[-int(k_size/2):k_size-int(k_size/2),-int(k_size/2):k_size-int(k_size/2)]
#  g_kernel=np.zeros((k_size,k_size))
#  g_kernel=np.exp(-(kx**2+ky**2)/(2*sigma**2))
#  g_kernel /=np.sum(g_kernel)
#  return(g_kernel)
# def Filtering(img,filter_x):
#     out_image=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
#     g_kernel=filter_x
#     for y in range(int(filter_x.shape[0]/2),img.shape[0]-int(filter_x.shape[1]/2)):
#         for x in range(int(filter_x.shape[0]/2),img.shape[1]-int(filter_x.shape[1]/2)):
#             sub_image=img[y-int(filter_x.shape[0]/2):y+int(filter_x.shape[0]/2)+1,x-int(filter_x.shape[1]/2):x+int(filter_x.shape[1]/2)+1]
#             sub_image1=g_kernel*sub_image
#             out_image[y,x]=np.sum(sub_image1)
#     return out_image

# a=Bilateral_filter(out_img,5,2,2)
# print(np.count_nonzero(a==255))
# print(np.count_nonzero(a==0))
# f=Gaussian_filter(5,3)
# b=Filtering(out_img,f)
# print(np.count_nonzero(a==255))
# print(np.count_nonzero(a==0))
# def MSE(img1,img2):
#     MSE=np.sum((img1-img2)**2)/(img1.size)
#     return MSE
# def PSNR(img1,img2):
#     M=MSE(img1,img2)
#     PSNR=10*np.log10((255*255)/M)
#     return PSNR
# print(MSE(img,out_img))
# print(MSE(out_img,a))
# print(MSE(out_img,b))
# print(MSE(a,b))
# print(PSNR(img,out_img))
# print(PSNR(out_img,a))
# print(PSNR(out_img,b))
# print(PSNR(a,b))
import numpy as np
ls1=[[100,200,1],[50,100,5],[25,50,9]]
ls2=[[200,400,1],[80,150,5],[30,120,9]]
ls3=[[400,200,1],[100,50,5],[200,100,9]]
ls=ls1+ls2+ls3
ls_np=np.array(ls)
