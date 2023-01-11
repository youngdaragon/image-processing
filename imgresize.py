import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

for i in range(5):
  keyword=i+1
  url=f'../Image/{keyword}.jpg'
  img = cv2.imread(url, cv2.IMREAD_COLOR)
  img=cv2.resize(img,dsize=(1920,1080),interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(f'{keyword}.jpg',img)

# keyword=5
# url=f'../Image/{keyword}.jpg'
# img = cv2.imread(url, cv2.IMREAD_COLOR)
# img=cv2.resize(img,dsize=(1920,1080),interpolation=cv2.INTER_CUBIC)
# cv2.imwrite(f'{keyword}.jpg',img)