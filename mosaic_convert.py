import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('../Image/lena.bmp', cv2.IMREAD_COLOR)

def mosaicked_img1(color_img):
    converted_img=np.zeros((color_img.shape[0],color_img.shape[1]),dtype=np.uint8)
    R=color_img[:,:,0]
    G=color_img[:,:,1]
    B=color_img[:,:,2]

    for i in range(color_img.shape[0]//2):
        for j in range(color_img.shape[1]//2):
            converted_img[2*i][2*j+1]=R[2*i][2*j+1]
            converted_img[2*i+1][2*j]=B[2*i+1][2*j]
            converted_img[2*i+1][2*j+1]=G[2*i+1][2*j+1]
            converted_img[2*i][2*j]=G[2*i][2*j]
    return converted_img

def mosaicked_img2(color_img):
    converted_img=np.zeros((color_img.shape[0],color_img.shape[1]),dtype=np.uint8)
    R=color_img[:,:,0]
    G=color_img[:,:,1]
    B=color_img[:,:,2]

    for i in range(color_img.shape[0]//2):
        for j in range(color_img.shape[1]//2):
            if (j%2)==1:
                converted_img[2*i][2*j+1]=R[2*i][2*j+1]
                converted_img[2*i+1][2*j+1]=B[2*i+1][2*j+1]
            else:
                converted_img[2*i+1][2*j+1]=R[2*i+1][2*j+1]
                converted_img[2*i][2*j+1]=B[2*i][2*j+1]
            converted_img[2*i+1][2*j]=G[2*i+1][2*j]
            converted_img[2*i][2*j]=G[2*i][2*j]
    return converted_img

def mosaicked_img3(color_img):
    converted_img=np.zeros((color_img.shape[0],color_img.shape[0]),dtype=np.uint8)
    R=color_img[:,:,0]
    G=color_img[:,:,1]
    B=color_img[:,:,2]

    for i in range(color_img.shape[0]//2):
        for j in range(color_img.shape[0]//2):
            if (i%2)==1:
                converted_img[2*i][2*j+1]=G[2*i][2*j+1]
                converted_img[2*i+1][2*j]=B[2*i+1][2*j]
                converted_img[2*i+1][2*j+1]=G[2*i+1][2*j+1]
                converted_img[2*i][2*j]=R[2*i][2*j]
            else:
                converted_img[2*i][2*j+1]=R[2*i][2*j+1]
                converted_img[2*i+1][2*j]=G[2*i+1][2*j]
                converted_img[2*i+1][2*j+1]=B[2*i+1][2*j+1]
                converted_img[2*i][2*j]=G[2*i][2*j]
    return converted_img

def mosaicked_img4(color_img):
    converted_img=np.zeros((color_img.shape[0],color_img.shape[0]),dtype=np.uint8)
    R=color_img[:,:,0]
    G=color_img[:,:,1]
    B=color_img[:,:,2]

    for i in range(color_img.shape[0]):
        for j in range(color_img.shape[0]):
            if (j%3)==0:
                converted_img[i][j]=G[i][j]
            elif (j%3)==1:
                converted_img[i][j]=R[i][j]
            else:
                converted_img[i][j]=B[i][j]
    return converted_img


def demosaicked_img1(converted_img):
    img=np.float32(converted_img)
    image_r=np.zeros(img.shape)
    image_b=np.zeros(img.shape)
    image_g=np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(y%2==0 and x%2==1) or (y==img.shape[0]-1) or (x==0):   
                image_r[y,x]=img[y,x]
            elif(y%2==1 and x%2==0):
                image_r[y,x]=(img[y-1,x-1]+img[y+1,x-1]+img[y-1,x+1]+img[y+1,x+1])//4
            elif(y%2==0 and x%2==0):
                image_r[y,x]=(img[y,x-1]+img[y,x+1])//2
            elif(y%2==1 and x%2==1):
                image_r[y,x]=(img[y-1,x]+img[y+1,x])//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(y%2==1 and x%2==0) or (y==0) or (x==img.shape[0]-1):   
                image_b[y,x]=img[y,x]
            elif(y%2==0 and x%2==1):
                image_b[y,x]=(img[y-1,x-1]+img[y+1,x-1]+img[y-1,x+1]+img[y+1,x+1])//4
            elif(y%2==0 and x%2==0):
                image_b[y,x]=(img[y-1,x]+img[y+1,x])//2
            elif(y%2==1 and x%2==1):
                image_b[y,x]=(img[y,x+1]+img[y,x-1])//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(y==img.shape[0]-1) and (x==0):
                image_g[y,x]=(img[y-1,x]+img[y,x+1])//2
            elif(y==0) and (x==img.shape[1]-1):
                image_g[y,x]=(img[y+1,x]+img[y,x-1])//2
            elif(y==img.shape[0]-1) and (x==img.shape[1]-1):
                image_g[y,x]=(img[y-1,x]+img[y,x-1])//2
            elif(y==0 and x%2==1):
                image_g[y,x]=(img[y,x+1]+img[y,x-1]+img[y+1,x])//3
            elif(y%2==1 and x==0):
                image_g[y,x]=(img[y-1,x]+img[y+1,x]+img[y,x+1])//3
            elif(y==img.shape[0]-1 and x%2==1):
                image_g[y,x]=(img[y,x+1]+img[y,x-1]+img[y-1,x])//3
            elif(y%2==1 and x==img.shape[1]-1):
                image_g[y,x]=(img[y-1,x]+img[y+1,x]+img[y,x-1])//3
            elif(y%2==1 and x%2==1) or (y==img.shape[0]-1) or (x==0):
                image_g[y,x]=img[y,x]
            elif(y%2==0 and x%2==0) or (y==0) or (x==img.shape[0]-1):
                image_g[y,x]=img[y,x]
            elif(y%2==0 and x%2==1):
                image_g[y,x]=(img[y,x+1]+img[y+1,x]+img[y-1,x]+img[y,x-1])//4
            elif(y%2==1 and x%2==0):   
                image_g[y,x]=(img[y,x+1]+img[y+1,x]+img[y-1,x]+img[y,x-1])//4

    convert_img=np.zeros((512,512,3))
    convert_img[:,:,0]=image_r
    convert_img[:,:,1]=image_g
    convert_img[:,:,2]=image_b
    convert_img=np.uint8(convert_img)
    return convert_img

def demosaicked_img2(converted_img):
    img=np.float32(converted_img)
    image_r=np.zeros(img.shape)
    image_b=np.zeros(img.shape)
    image_g=np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%4==1 and y%2==0) or (y==img.shape[0]-1) or (x==0):   
                image_b[y,x]=img[y,x]
            elif(y%2==1 and x%4==3) or (y==0):
                image_b[y,x]=img[y,x]
            elif(x%4==0 and y%2==0):
                image_b[y,x]=img[y,x+1]
            elif(y%2==1 and x%4==2):
                image_b[y,x]=img[y,x+1]
            elif(y%2==0 and x%4==2):
                image_b[y,x]=img[y,x-1]
            elif(y%2==1 and x%4==0):
                image_b[y,x]=img[y,x-1]     
            elif(y%2==0 and x%4==3):
                image_b[y,x]=(img[y-1,x]+img[y+1,x])//2
            elif(y%2==1 and x%4==1):
                image_b[y,x]=(img[y-1,x]+img[y+1,x])//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%4==1 and y%2==1) or (y==0) or (x==0):   
                image_r[y,x]=img[y,x]
            elif(y%2==0 and x%4==3) or (y==img.shape[0]-1):
                image_r[y,x]=img[y,x]
            elif(x%4==2 and y%2==0):
                image_r[y,x]=img[y,x+1]
            elif(y%2==1 and x%4==0):
                image_r[y,x]=img[y,x+1]
            elif(y%2==1 and x%4==2):
                image_r[y,x]=img[y,x-1]
            elif(y%2==0 and x%4==0):
                image_r[y,x]=img[y,x-1]
            elif(y%2==0 and x%4==1):
                image_r[y,x]=(img[y-1,x]+img[y+1,x])//2
            elif(y%2==1 and x%4==3):
                image_r[y,x]=(img[y-1,x]+img[y+1,x])//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%2==0) or (x==img.shape[0]-1):
                image_g[y,x]=img[y,x]
            elif(x%2==1):
                image_g[y,x]=(img[y,x+1]+img[y,x-1])//2
            

    convert_img=np.zeros((512,512,3))
    convert_img[:,:,0]=image_r
    convert_img[:,:,1]=image_g
    convert_img[:,:,2]=image_b
    convert_img=np.uint8(convert_img)
    return convert_img

def demosaicked_img3(converted_img):
    img=np.float32(converted_img)
    image_r=np.zeros(img.shape)
    image_b=np.zeros(img.shape)
    image_g=np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%2==1 and y%4==0) or (y==img.shape[0]-1) or (x==0):   
                image_r[y,x]=img[y,x]
            elif(x%2==0 and y%4==2) or (y==0) or (x==img.shape[1]-1):
                image_r[y,x]=img[y,x]
            elif(y%4==0 and x%2==0):
                image_r[y,x]=(img[y,x+1]+img[y,x-1])//2
            elif(y%4==2 and x%2==1):
                image_r[y,x]=(img[y,x+1]+img[y,x-1])//2
            elif(x%2==1 and y%4==1):
                image_r[y,x]=img[y-1,x]
            elif(x%2==0 and y%4==1):
                image_r[y,x]=img[y+1,x]     
            elif(x%2==0 and y%4==3):
                image_r[y,x]=img[y-1,x]
            elif(x%2==1 and y%4==3):
                image_r[y,x]=img[y+1,x]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(y%4==1 and x%2==1) or (y==img.shape[0]-1) or (x==0):   
                image_b[y,x]=img[y,x]
            elif(y%4==3 and x%2==0) or (y==0) or (x==img.shape[1]-1):
                image_b[y,x]=img[y,x]
            elif(y%4==0 and x%2==0):
                image_b[y,x]=img[y-1,x]
            elif(y%4==0 and x%2==1):
                image_b[y,x]=img[y+1,x]
            elif(y%4==1 and x%2==0):
                image_b[y,x]=(img[y,x-1]+img[y,x+1])//2
            elif(y%4==2 and x%2==0):
                image_b[y,x]=img[y+1,x]     
            elif(y%4==2 and x%2==1):
                image_b[y,x]=img[y-1,x]
            elif(y%4==3 and x%2==1):
                image_b[y,x]=(img[y,x+1]+img[y,x-1])//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%2==0 and y%4==0) or (x==img.shape[1]-1) or (y==0) or (y==img.shape[0]-1)or (x==0):
                image_g[y,x]=img[y,x]
            elif(x%2==0 and y%4==1):
                image_g[y,x]=img[y,x]
            elif(x%2==1 and y%4==2):
                image_g[y,x]=img[y,x]
            elif(x%2==1 and y%4==3):
                image_g[y,x]=img[y,x]
            elif(x%2==0):
                if(y%4==3):
                    image_g[y,x]=(img[y+1,x]+img[y,x-1]+img[y,x+1])//3
                if(y%4==2):
                    image_g[y,x]=(img[y-1,x]+img[y,x-1]+img[y,x+1])//3
            elif(x%2==1):
                if(y%4==0):
                    image_g[y,x]=(img[y-1,x]+img[y,x-1]+img[y,x+1])//3
                if(y%4==1):
                    image_g[y,x]=(img[y+1,x]+img[y,x-1]+img[y,x+1])//3
    convert_img=np.zeros((512,512,3))
    convert_img[:,:,0]=image_r
    convert_img[:,:,1]=image_g
    convert_img[:,:,2]=image_b
    convert_img=np.uint8(convert_img)
    return convert_img

def demosaicked_img4(converted_img):
    img=np.float32(converted_img)
    image_r=np.zeros(img.shape)
    image_b=np.zeros(img.shape)
    image_g=np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%3==0):   
                image_r[y,x]=img[y,x+1]
            elif(x%3==1) or (x==0) or (x==img.shape[1]-1):
                image_r[y,x]=img[y,x]
            elif(x%3==2):
                image_r[y,x]=img[y,x-1]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%3==2) or (x==0) or (x==img.shape[1]-1):
                image_b[y,x]=img[y,x]
            elif(x%3==0):   
                image_b[y,x]=img[y,x-1]
            elif(x%3==1):
                image_b[y,x]=img[y,x+1]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(x%3==0) or (x==0) or (x==1):
                image_g[y,x]=img[y,x]
            elif(x%3==1):
                image_g[y,x]=img[y,x-1]
            elif(x%3==2):
                image_g[y,x]=img[y,x+1]
            
        
    convert_img=np.zeros((512,512,3))
    convert_img[:,:,0]=image_r
    convert_img[:,:,1]=image_g
    convert_img[:,:,2]=image_b
    convert_img=np.uint8(convert_img)
    return convert_img

def MSE(img1,img2):
    MSE=np.sum((img1-img2)**2)/(img1.size)
    return MSE

def PSNR(img1,img2):
    M=MSE(img1,img2)
    average_MSE=M/3
    PSNR=10*np.log10((255*255)/average_MSE)
    return PSNR

converted_img1=mosaicked_img1(img)
converted_img2=mosaicked_img2(img)
converted_img3=mosaicked_img3(img)
converted_img4=mosaicked_img4(img)
converted_img5=0.3*img[:,:,0]+0.5*img[:,:,1]+0.2*img[:,:,2]
# plt.imshow(converted_img4,cmap='gray')
# plt.show()
convert_img1=demosaicked_img1(converted_img1)
convert_img2=demosaicked_img2(converted_img2)
convert_img3=demosaicked_img3(converted_img3)
convert_img4=demosaicked_img4(converted_img4)
print('a패턴 복원한 결과: ',PSNR(img,convert_img1))
print('b패턴 복원한 결과: ',PSNR(img,convert_img2))
print('c패턴 복원한 결과: ',PSNR(img,convert_img3))
print('d패턴 복원한 결과: ',PSNR(img,convert_img4))
fix_img1 = cv2.cvtColor(convert_img1, cv2.COLOR_BGR2RGB)
fix_img2 = cv2.cvtColor(convert_img2, cv2.COLOR_BGR2RGB)
fix_img3 = cv2.cvtColor(convert_img3, cv2.COLOR_BGR2RGB)
fix_img4 = cv2.cvtColor(convert_img4, cv2.COLOR_BGR2RGB)
fix_img5 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(3,2,1)
plt.title('a')
plt.imshow(fix_img1,cmap='gray')
plt.subplot(3,2,2)
plt.title('b')
plt.imshow(fix_img2,cmap='gray')
plt.subplot(3,2,3)
plt.title('c')
plt.imshow(fix_img3,cmap='gray')
plt.subplot(3,2,4)
plt.title('d')
plt.imshow(fix_img4,cmap='gray')
plt.subplot(3,2,5)
plt.title('GT_img')
plt.imshow(img,cmap='gray')
plt.subplot(3,2,6)
plt.title('gray_img')
plt.imshow(fix_img5)
plt.show()





    

