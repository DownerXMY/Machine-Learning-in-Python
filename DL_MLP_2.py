import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

from sklearn.datasets import load_iris
iris = load_iris()
data_X = np.array(iris['data'])
data_Y = np.array(iris['target'])
print(data_X.shape,data_Y.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
data_X1 = pca.fit_transform(data_X)
print(pca.explained_variance_ratio_)

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
data_Y1 = LabelBinarizer().fit_transform(data_Y)
model = Sequential(
    [
        Dense(3,input_dim=3),
        Activation('relu'),
        Dense(3),
        Activation('sigmoid')
    ]
)
adam = Adam(learning_rate=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy')
model.fit(data_X1,data_Y1,batch_size=40,epochs=500)
result = model.predict_classes(data_X1)

X1_min,X1_max = data_X1[:,0].min()-1,data_X1[:,0].max()+1
X2_min,X2_max = data_X1[:,1].min()-1,data_X1[:,1].max()+1
X3_min,X3_max = data_X1[:,2].min()-1,data_X1[:,2].max()+1
XX1,XX2,XX3 = np.meshgrid(np.arange(X1_min,X1_max,0.02),
                          np.arange(X2_min,X2_max,0.02),
                          np.arange(X3_min,X3_max,0.02))
Y_mesh = model.predict_classes(np.array([XX1.ravel(),XX2.ravel(),XX3.ravel()]).T)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['red','blue','green'])

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import subplot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(121,projection='3d')
ax.scatter(data_X1[:,0],data_X1[:,1],data_X1[:,2],marker='o',color='black')
ax1 = fig.add_subplot(122,projection='3d')
ax1.contourf(XX1,XX2,XX3,Y_mesh.reshape(XX1.shape),cmap=cmap,alpha=0.4)
ax1.scatter(data_X1[result==0,0],data_X1[result==0,1],data_X1[result==0,2],marker='o',color='red')
ax1.scatter(data_X1[result==1,0],data_X1[result==1,1],data_X1[result==1,2],marker='o',color='blue')
ax1.scatter(data_X1[result==2,0],data_X1[result==2,1],data_X1[result==2,2],marker='o',color='green')
plt.show()
