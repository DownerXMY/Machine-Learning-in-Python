import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

file_DY = pd.read_csv('/Users/mingyuexu/PycharmProjects/demo/16区抖音采集(1).csv')
# print(file_DY)
# print([column for column in file_DY])
file_DY = file_DY.drop(columns=['Unnamed: 8','Unnamed: 9','Unnamed: 10'])
print(file_DY,file_DY.shape)
data_X = file_DY.drop(columns=['账号名','清博DCI'])
data_X = (data_X-np.mean(data_X))/np.std(data_X)
data_Y = file_DY.loc[:,'清博DCI']
data_Y = (data_Y-np.mean(data_Y))/np.std(data_Y)
# print(data_X,data_Y)
from sklearn.decomposition import PCA
pca1 = PCA(n_components=7)
pca1.fit_transform(data_X)
print(pca1.explained_variance_ratio_)
pca = PCA(n_components=5)
data_X1 = pca.fit_transform(data_X)
print(data_X1.shape)

from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential(
    [
        Dense(6,input_dim=5),
        Activation('relu'),
        Dense(1),
        Activation('relu')
    ]
)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(data_X1,data_Y,epochs=5000)
result1 = model.predict(data_X1)
result1_rebuild = result1 * np.std(data_Y) + np.mean(data_Y)
from sklearn.metrics import mean_squared_error
print('loss:',mean_squared_error(y_true=data_Y,y_pred=result1))

fig1 = plt.figure()
from matplotlib.pyplot import subplot
subplot(121)
plt.scatter(np.arange(16),data_Y)
plt.plot(data_Y)
plt.title('desired result')
subplot(122)
plt.scatter(np.arange(16),result1_rebuild)
plt.plot(result1_rebuild)
plt.title('test result')
plt.show()

fig2 = plt.figure()
plt.scatter(np.arange(16),data_Y,color='blue')
plt.scatter(np.arange(16),result1_rebuild,color='red')
plt.plot(data_Y,label='desired result')
plt.plot(result1_rebuild,label='test result')
plt.ylabel('DCI')
plt.xlabel('rank')
plt.legend()
plt.show()
