import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

print('The progress of the CNN-Net:')
print('AlexNet --> VGGNet --> InceptionNet/ResNet --> InceptionResNet --> NASNet --> MobileNet')
print('-----------------------------------------')

print('ResNet:')
from keras.applications.resnet import preprocess_input,ResNet50
model_Res = ResNet50(include_top=False,weights='imagenet')
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import preprocess_input as p1
model_NasMobile = NASNetMobile(include_top=False,weights='imagenet')

from keras.preprocessing.image import load_img,img_to_array
import os
path_cat = '/Users/mingyuexu/PycharmProjects/vgg_data_cat_c'
img_name_cat = os.listdir(path_cat)
data_cat = []
data_dog = []
data_cat1 = []
data_dog1 = []
for item in img_name_cat:
    img = load_img(f'/Users/mingyuexu/PycharmProjects/vgg_data_cat_c/{item}',target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    X = preprocess_input(img_array)
    X1 = p1(img_array)
    data_cat.append(X)
    data_cat1.append(X1)
data_cat = np.array(data_cat)
data_cat1 = np.array(data_cat1)
# print(data_cat.shape)

path_dog = '/Users/mingyuexu/PycharmProjects/vgg_data_dog_c'
img_name_dog = os.listdir(path_dog)
for item in img_name_dog:
    img = load_img(f'/Users/mingyuexu/PycharmProjects/vgg_data_dog_c/{item}',target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    X = preprocess_input(img_array)
    X1 = p1(img_array)
    data_dog.append(X)
    data_dog1.append(X1)
data_dog = np.array(data_dog)
data_dog1 = np.array(data_dog1)

data_ori = np.concatenate((data_cat,data_dog),axis=0)
data_ori1 = np.concatenate((data_cat1,data_dog1),axis=0)
# print(data_ori.shape)
data_X = np.zeros(7*7*2048)
for item in data_ori:
    data1 = model_Res.predict(item)
    # print(data1.shape)
    data1 = data1.reshape(1,7*7*2048)
    data_X = np.vstack((data_X,data1))
data_X = data_X[1:,:]
# print(data_X.shape)
data_Y = np.hstack((np.zeros(200),np.ones(200)))

data_X1 = np.zeros(7*7*1056)
for item in data_ori1:
    data3 = model_NasMobile.predict(item)
    # print(data3.shape)
    data3 = data3.reshape(1,7*7*1056)
    data_X1 = np.vstack((data_X1,data3))
data_X1 = data_X1[1:,:]
# print(data_X.shape)
data_Y1 = np.hstack((np.zeros(200),np.ones(200)))

from keras.models import Sequential
from keras.layers import Dense,Activation

model_after_Res = Sequential(
    [
        Dense(10,input_dim=100352),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model_after_Res.compile(optimizer='adam',loss='binary_crossentropy')
model_after_Res.fit(data_X,data_Y,epochs=50)
result_Res = model_after_Res.predict_classes(data_X)
from sklearn import metrics
print('accuracy for ResNet:',metrics.accuracy_score(y_true=data_Y,y_pred=result_Res))

path_cat_test = '/Users/mingyuexu/PycharmProjects/vgg_cat_test'
img_name_cat_test = os.listdir(path_cat_test)
data_cat_test = []
data_dog_test = []
data_cat_test1 = []
data_dog_test1 = []
for item in img_name_cat_test:
    img = load_img(f'/Users/mingyuexu/PycharmProjects/vgg_cat_test/{item}',target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    X = preprocess_input(img_array)
    X1 = p1(img_array)
    data_cat_test.append(X)
    data_cat_test1.append(X1)
data_cat_test = np.array(data_cat_test)
data_cat_test1 = np.array(data_cat_test1)

path_dog_test = '/Users/mingyuexu/PycharmProjects/vgg_dog_test'
img_name_dog_test = os.listdir(path_dog_test)
for item in img_name_dog_test:
    img = load_img(f'/Users/mingyuexu/PycharmProjects/vgg_dog_test/{item}',target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    X = preprocess_input(img_array)
    X1 = p1(img_array)
    data_dog_test.append(X)
    data_dog_test1.append(X1)
data_dog_test = np.array(data_dog_test)
data_dog_test1 = np.array(data_dog_test1)
data_test_raw = np.concatenate((data_cat_test,data_dog_test),axis=0)
data_test_raw1 = np.concatenate((data_cat_test1,data_dog_test1),axis=0)

data_test = np.zeros(7*7*2048)
for item in data_test_raw:
    data2 = model_Res.predict(item)
    data2 = data2.reshape(1,7*7*2048)
    data_test = np.vstack((data_test,data2))
data_test = data_test[1:,:]
data_test_label = np.hstack((np.zeros(100),np.ones(100)))
result_Res_test = model_after_Res.predict_classes(data_test)
print('accuracy for test data:',metrics.accuracy_score(y_true=data_test_label,y_pred=result_Res_test))
print('-----------------------------------------')
print('NASNet:')
model_after_NAS = Sequential(
    [
        Dense(10,input_dim=51744),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model_after_NAS.compile(optimizer='adam',loss='binary_crossentropy')
model_after_NAS.fit(data_X1,data_Y1,epochs=50)
result_NAS = model_after_NAS.predict_classes(data_X1)
print('accuracy for NASNet:',metrics.accuracy_score(y_true=data_Y1,y_pred=result_NAS))

data_test1 = np.zeros(7*7*1056)
for item in data_test_raw1:
    X2 = model_NasMobile.predict(item)
    X2 = X2.reshape(1,7*7*1056)
    data_test1 = np.vstack((data_test1,X2))
data_test1 = data_test1[1:,:]
data_test1_label = np.hstack((np.zeros(100),np.ones(100)))
result_NAS_test = model_after_NAS.predict_classes(data_test1)
print('accuracy for test data:',metrics.accuracy_score(y_true=data_test1_label,y_pred=result_NAS_test))
