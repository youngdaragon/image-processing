import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd
from sklearn import metrics
data=pd.read_csv('../data_labeling.csv',encoding='UTF-8')
data=data.to_numpy()
first=np.array([25,79,1])
data=np.vstack([first,data])
x=data[:,0:2]
y=data[:,2]
print(x.shape)
print(y.shape)
# clf=svm.SVC(kernel='linear',C=1)
# clf2=svm.SVC(kernel='rbf', gamma=0.7, C=1)
# clf2.fit(x,y)

# Yte_pred_linear_c1 = clf2.predict(x)

# print("Classification report for - \n{}:\n{}\n".format(Yte_pred_linear_c1, metrics.accuracy_score(y, Yte_pred_linear_c1)))

# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy


# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# # 모델 정의&피팅
# C = 1.0 #regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#           svm.LinearSVC(C=C, max_iter=10000),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C),
#           svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
# models = (clf.fit(x, y) for clf in models)

# # plot title 형성
# titles = ('SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel')

# fig, sub = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# X0, X1 = x[:, 0], x[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# for clf, title, ax in zip(models, titles, sub.flatten()):
#     plot_contours(ax, clf, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title)

# plt.show()


# a=np.random.rand(x.shape[1],1)
# y=y.reshape(1,-1)
# output=np.dot(x,a)-y.T
# print(output)
# output=np.sum(output)
# print(output)
