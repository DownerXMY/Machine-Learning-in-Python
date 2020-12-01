import numpy as np
from matplotlib import pyplot as plt

# Please run t1.py in advance.

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

X = []
for item in range(1,1001,1):
    img_path = f'/Users/mingyuexu/PycharmProjects/data_vgg/data_vgg_cat/cat.{item-1}的副本.jpg'
    img = load_img(img_path,target_size=(224,224))
    img = img_to_array(img)
    x = np.expand_dims(img,axis=0)
    x = preprocess_input(x)
    X.append(x)

for item in range(1,1001,1):
    img_path = f'/Users/mingyuexu/PycharmProjects/data_vgg/data_vgg_dog/dog.{item-1}的副本.jpg'
    img = load_img(img_path,target_size=(224,224))
    img = img_to_array(img)
    x = np.expand_dims(img,axis=0)
    x = preprocess_input(x)
    X.append(x)

data_X = np.array(X)
data_Y = np.hstack((np.zeros(1000),np.ones(1000)))
print(data_X.shape)
print(data_Y)

model_vgg = VGG16(weights='imagenet',include_top=False)
Feature = []
for x in data_X:
    features = model_vgg.predict(x)
    # print(features.shape)
    features = features.reshape(1, 7 * 7 * 512)
    Feature.append(features)

Feature = np.array(Feature)
# print(Feature.shape)

from keras.models import Sequential
from keras.layers import Dense,Activation
model_mlp = Sequential(
    [
        Dense(10,input_dim=25088),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model_mlp.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_mlp.fit(Feature,data_Y,epochs=50)
result = model_mlp.predict_classes(Feature)
# print(result)
n = 0
result_up = []
for item in result:
    if item == [[0]]:
        result_up.append(0)
    else:
        result_up.append(1)
    n += 1
print(result_up)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result_up))
