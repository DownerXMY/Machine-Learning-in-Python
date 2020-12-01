import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 5

# unsupervised learning
# clustering
print('clustering algorithms: KMeans, Meanshift, DBSCAN')
print('--------------------------------')

from sklearn.datasets import load_iris
iris = load_iris()
data_X = np.array(iris['data'])
data_Y = np.array(iris['target'])
print(data_X.shape)
# print(iris)
data_X = data_X[:,[2,3]]
# print(data_X)
label_0 = plt.scatter(data_X[data_Y==0,0],data_X[data_Y==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[data_Y==1,0],data_X[data_Y==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[data_Y==2,0],data_X[data_Y==2,1],marker='x',color='lightgreen')
plt.title('unlabeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.show()

# KMeans
print('KMeans analysis:')
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=3,random_state=0)
KM.fit(data_X)
result = KM.predict(data_X)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result))

# To examine why the accuracy is so low:
from matplotlib.pyplot import subplot
plt.subplot(121)
label_0 = plt.scatter(data_X[data_Y==0,0],data_X[data_Y==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[data_Y==1,0],data_X[data_Y==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[data_Y==2,0],data_X[data_Y==2,1],marker='x',color='lightgreen')
plt.title('unlabeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.subplot(122)
label_0 = plt.scatter(data_X[result==0,0],data_X[result==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[result==1,0],data_X[result==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[result==2,0],data_X[result==2,1],marker='x',color='lightgreen')
plt.title('labeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.show()

# To correct the prediction:
result_corrected = []
for item in result:
    if item == 0:
        result_corrected.append(0)
    if item == 1:
        result_corrected.append(2)
    if item == 2:
        result_corrected.append(1)
print('accuracy after corrected:',metrics.accuracy_score(y_true=data_Y,y_pred=result_corrected))

# To see the classfy result:
print(f'min_length={data_X[:,0].min()},max_length={data_X[:,0].max()},min_width={data_X[:,1].min()},max_width={data_X[:,1].max()}')
class_pred = KM.predict(np.array([[4.5,2.0]]))
label_0 = plt.scatter(data_X[data_Y==0,0],data_X[data_Y==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[data_Y==1,0],data_X[data_Y==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[data_Y==2,0],data_X[data_Y==2,1],marker='x',color='lightgreen')
plt.scatter(4.5,2.0,color='gray',marker='v')
for item in [0,1,2]:
    if class_pred == item:
        print(f'it belongs to label_{item}')
plt.title('unlabeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
centers = KM.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],color='black',marker='o')
plt.show()
print('--------------------------------')

# KNN
print('KNN analysis:')
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(data_X,data_Y)
result1 = KNN.predict(data_X)
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result1))
class_pred1 = KNN.predict(np.array([[4.5,2.0]]))
for item in [0,1,2]:
    if class_pred1 == item:
        print(f'it belongs to label_{item}')

plt.subplot(131)
label_0 = plt.scatter(data_X[data_Y==0,0],data_X[data_Y==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[data_Y==1,0],data_X[data_Y==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[data_Y==2,0],data_X[data_Y==2,1],marker='x',color='lightgreen')
plt.title('unlabeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.subplot(132)
label_0 = plt.scatter(data_X[result1==0,0],data_X[result1==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[result1==1,0],data_X[result1==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[result1==2,0],data_X[result1==2,1],marker='x',color='lightgreen')
plt.title('labeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))

# To show the demarcation:
X1_min,X1_max = data_X[:,0].min()-1,data_X[:,0].max()+1
X2_min,X2_max = data_X[:,1].min()-1,data_X[:,1].max()+1
XX1,XX2 = np.meshgrid(np.arange(X1_min,X1_max,0.01),
                      np.arange(X2_min,X2_max,0.01))
X_mesh = np.array([XX1.ravel(),XX2.ravel()]).T
result_mesh = KNN.predict(X_mesh)
colormap = ['red','blue','lightgreen']
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colormap)
subplot(133)
plt.contourf(XX1,XX2,result_mesh.reshape(XX1.shape),alpha=0.4,cmap=cmap)
plt.xlim(XX1.min(), XX1.max())
plt.ylim(XX2.min(), XX2.max())
label_0 = plt.scatter(data_X[result1==0,0],data_X[result1==0,1],marker='x',color='red',linewidths=1)
label_1 = plt.scatter(data_X[result1==1,0],data_X[result1==1,1],marker='x',color='blue',linewidths=1)
label_2 = plt.scatter(data_X[result1==2,0],data_X[result1==2,1],marker='x',color='lightgreen',linewidths=1)
plt.title('labeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.show()
print('---------------------------------------')

# Meanshift
print('Meanshift analysis:')
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(data_X,n_samples=50)
MS = MeanShift(bandwidth=bandwidth)
MS.fit(data_X)
result2 = MS.predict(data_X)
import pandas as pd
res2 = pd.Series(result2)
print(res2.value_counts())
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result2))

plt.subplot(121)
label_0 = plt.scatter(data_X[data_Y==0,0],data_X[data_Y==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[data_Y==1,0],data_X[data_Y==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[data_Y==2,0],data_X[data_Y==2,1],marker='x',color='lightgreen')
plt.title('unlabeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.subplot(122)
label_0 = plt.scatter(data_X[result2==0,0],data_X[result2==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[result2==1,0],data_X[result2==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[result2==2,0],data_X[result2==2,1],marker='x',color='lightgreen')
plt.title('labeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.show()

# Similarly, we need to correct the prediction:
result_corrected1 = []
for item in result2:
    if item == 0:
        result_corrected1.append(1)
    if item == 1:
        result_corrected1.append(0)
    if item == 2:
        result_corrected1.append(2)
print('accuracy after corrected:',metrics.accuracy_score(y_true=data_Y,y_pred=result_corrected1))
print('confusion matrix:',metrics.confusion_matrix(y_true=data_Y,y_pred=result_corrected1))
label_0 = plt.scatter(data_X[np.array(result_corrected1)==0,0],data_X[np.array(result_corrected1)==0,1],marker='x',color='red')
label_1 = plt.scatter(data_X[np.array(result_corrected1)==1,0],data_X[np.array(result_corrected1)==1,1],marker='x',color='blue')
label_2 = plt.scatter(data_X[np.array(result_corrected1)==2,0],data_X[np.array(result_corrected1)==2,1],marker='x',color='lightgreen')
centers1 = MS.cluster_centers_
plt.scatter(centers1[:,0],centers1[:,1],color='black',marker='x',linewidth=1)
plt.title('labeled data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend((label_0,label_1,label_2),('label_0','label_1','label_2'))
plt.show()
print('------------------------------------')