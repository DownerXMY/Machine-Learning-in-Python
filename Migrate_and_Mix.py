import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# transfer learning
# online learning
# semi-supervised learning
# pseudo label learning

print('transfer-learning:')
S1 = np.load('mnist.npz')
X_train,Y_train,X_test,Y_test = S1['x_train'],S1['y_train'],S1['x_test'],S1['y_test']
# print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
X_minst = np.concatenate((X_train,X_test),axis=0)
Y_minst = np.concatenate((Y_train,Y_test),axis=0)
print(X_minst.shape,Y_minst.shape)

import scipy.io as scio
usps_img = scio.loadmat('/Users/mingyuexu/PycharmProjects/demo/usps_train_16x16.mat')
X_usps_raw = usps_img['usps_train'].reshape(4649,16,16)
X_usps = []
for item in X_usps_raw:
    Xi = np.ones(784).reshape(28,28) * (-1)
    Xi[6:22,6:22] = item
    X_usps.append(Xi)
X_usps = np.array([X_usps]).reshape(4649,28,28)
usps_label = scio.loadmat('/Users/mingyuexu/PycharmProjects/demo/usps_train_labels_16x16.mat')
Y_usps_raw = usps_label['usps_train_labels'].reshape(4649,)
Y_usps = []
for item in Y_usps_raw:
    Y_usps.append(item-1)
Y_usps = np.array(Y_usps)
print(X_usps.shape,Y_usps.shape)

feature_size1 = X_minst[0].shape[0] * X_minst[0].shape[1]
X_minst_format = X_minst.reshape(X_minst.shape[0],feature_size1)
print(X_minst_format.shape)
X_minst_normal = X_minst_format/255
feature_size2 = X_usps[0].shape[0] * X_usps[0].shape[1]
X_usps_format = X_usps.reshape(X_usps.shape[0],feature_size2)
print(X_usps_format.shape)
X_usps_normal = X_usps_format/255

from matplotlib.pyplot import subplot
for item in range(1,10,1):
    subplot(3,3,item)
    # print(Y_usps[item-1])
    img = X_usps[item-1]
    plt.imshow(img)
plt.show()

from keras.utils import to_categorical
Y_minst_cato = to_categorical(Y_minst,num_classes=10)
Y_usps_cato = to_categorical(Y_usps,num_classes=10)

from keras.models import Sequential,load_model
from keras.layers import Dense,Activation
model = Sequential(
    [
        Dense(392,input_dim=784),
        Activation('relu'),
        Dense(392),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ]
)
model.compile(optimizer='adam',loss='categorical_crossentropy')
model.fit(X_minst_normal,Y_minst_cato,epochs=10)

# save the model
model.save('/Users/mingyuexu/PycharmProjects/demo/learning/model1.h5')

result1 = model.predict_classes(X_minst_normal)
from sklearn import metrics
print('accuracy for mnist:',metrics.accuracy_score(y_true=Y_minst,y_pred=result1))
print('-----------------------------------------')
print('start our transfer-learning:')

model1 = load_model('/Users/mingyuexu/PycharmProjects/demo/learning/model1.h5')
model1.fit(X_usps_normal,Y_usps_cato,epochs=10)
result2 = model1.predict_classes(X_usps_normal)
print('accuracy after transfering:',metrics.accuracy_score(y_true=Y_usps,y_pred=result2))

print('-----------------------------------------')
print('mixing learning:')
# please run t1.py firstly.

# We need to enlarge the data since its size given is only 10.
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
path = '/Users/mingyuexu/PycharmProjects/data_apple'
update_path = '/Users/mingyuexu/PycharmProjects/data_apple_update'
data_generator = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.02,horizontal_flip=True,vertical_flip=True)
data_update = data_generator.flow_from_directory(path,target_size=(224,224),batch_size=2,save_to_dir=update_path,save_prefix='up',save_format='jpg')
for item in range(100):
    data_update.next()

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
model_VGG = VGG16(weights='imagenet',include_top=False)

import os
folder = '/Users/mingyuexu/PycharmProjects/data_apple_update'
dirs = os.listdir(folder)

X_train_apple = np.array([np.zeros(25088)])
for item in dirs:
    img = load_img(f'/Users/mingyuexu/PycharmProjects/data_apple_update/{item}',target_size=(224,224))
    img_data = img_to_array(img)
    X = np.expand_dims(img_data,axis=0)
    X = preprocess_input(X)
    feature = model_VGG.predict(X)
    feature = feature.reshape(1,7*7*512)
    X_train_apple = np.vstack((X_train_apple,feature))
X_train_apple = X_train_apple[1:,:]
print(X_train_apple.shape)

# apply the KMeans analysis
print('KMeans:')
from sklearn.cluster import KMeans
model_KM = KMeans(n_clusters=2)
model_KM.fit(X_train_apple)
result3 = model_KM.predict(X_train_apple)
print(result3)
import pandas as pd
result3_up = pd.Series(result3)
print(result3_up.value_counts())
print('This is not the result desired!')

from matplotlib.pyplot import subplot
n = 1
for item in dirs:
    subplot(5,5,n)
    plt.imshow(load_img(f'/Users/mingyuexu/PycharmProjects/data_apple_update/{item}',target_size=(224,224)))
    plt.title(f'{result3[n-1]}')
    n += 1
    if n >= 26:
        break
plt.show()
print('----------------------------------------')
print('Meanshift:')
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(X_train_apple,n_samples=100)
model_MS = MeanShift(bandwidth=bandwidth)
model_MS.fit(X_train_apple)
result4 = model_MS.predict(X_train_apple)
result4_up = pd.Series(result4)
print(result4_up.value_counts())
print('that is wonderful!!!')

m = 1
k = 0
for item in result4:
    k += 1
    if item != 0:
        subplot(4,4,m)
        m += 1
        plt.imshow(load_img(f'/Users/mingyuexu/PycharmProjects/data_apple_update/{dirs[k-1]}',target_size=(224,224)))
plt.show()
print('-----------------------------------------')
# Can we improve the Meanshift model?

print('PCA before Meanshift:')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=229)
X_train_apple_std = StandardScaler().fit_transform(X_train_apple)
X_train_apple_up = pca.fit_transform(X_train_apple_std)
# print(pca.explained_variance_ratio_)
bandwidth_up = estimate_bandwidth(X_train_apple_up,n_samples=100)
model_MS_up = MeanShift(bandwidth=bandwidth_up)
model_MS_up.fit(X_train_apple_up)
result5 = model_MS_up.predict(X_train_apple_up)
result5_up = pd.Series(result5)
print(result5_up.value_counts())

p = 1
q = 0
for item in result5:
    q += 1
    if item != 0:
        subplot(6,6,p)
        p += 1
        plt.imshow(load_img(f'/Users/mingyuexu/PycharmProjects/data_apple_update/{dirs[p-1]}',target_size=(224,224)))
plt.show()
