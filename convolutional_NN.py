import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

from keras.preprocessing.image import ImageDataGenerator
train_data_generator = ImageDataGenerator(rescale=1.0/255)
train_data = train_data_generator.flow_from_directory('/Users/mingyuexu/PycharmProjects/train',target_size=(50,50),batch_size=32,class_mode='binary')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation

model = Sequential(
    [
        Conv2D(32,(3,3),input_shape=(50,50,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32,(3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data,epochs=25)
print(train_data.class_indices)
print('accuracy on training data:',model.evaluate(train_data)[1])

# test_data = train_data_generator.flow_from_directory('/Users/mingyuexu/PycharmProjects/test1',target_size=(50,50),batch_size=32,class_mode='binary')
# print('accuracy on test data:',model.evaluate(test_data)[1])

from keras.preprocessing.image import load_img,img_to_array
test_eg = '/Users/mingyuexu/PycharmProjects/cat_example.jpg'
test_eg = load_img(test_eg,target_size=(50,50))
test_eg = img_to_array(test_eg)
test_eg = test_eg/255
test_eg = test_eg.reshape(1,50,50,3)
result = model.predict_classes(test_eg)
if result == 1:
    print('cat')
else:
    print('dog')

