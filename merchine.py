import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd
from sklearn import metrics
import cv2
data = cv2.imread('../Image/5.jpg')
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
label=cv2.imread('../Image/GT_label.jpg',cv2.IMREAD_GRAYSCALE)

def svm_Kernel(data,label):
    x=data.reshape(-1,3)
    y=label.reshape(-1)

    clf=svm.SVC(kernel='linear',C=1)
    clf2=svm.SVC(kernel='rbf', gamma=1, C=1)
    clf3=svm.SVC(kernel='poly', degree=4, gamma='auto', C=1)
    clf4=svm.LinearSVC(C=1, max_iter=10000)
    clf3.fit(x,y)

    Yte_pred_linear_c1 = clf3.predict(x)

    Yte_pred_linear_c1=Yte_pred_linear_c1.reshape(data.shape[0],data.shape[1])
    cv2.imwrite('../Image/SVM_label.jpg',Yte_pred_linear_c1)

    return Yte_pred_linear_c1

svm_Kernel(data,label)