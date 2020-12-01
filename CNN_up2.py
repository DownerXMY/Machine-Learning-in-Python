import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

print('Optimizer:')
print('对于稀疏数据，一般选择学习率可自适应的方法','SGD通常学习时间更长，但是效果比较好，需要好的初始化和lr','较深较复杂的网络推荐adam',sep='\n')
print('---------------------------------')
print('Activation:')
print('不建议使用sigmoid','建议使用Leaky Relu,maxout,elu','可以尝试tanh,但不要抱太大希望')
print('---------------------------------')
print('Weights initialization:')
print('对于CNN而言,一种比较好的初始化方法是：')
print('W = np.random.randn(fan_in,fan_out)/np.sqrt(fan_in),其中fan_in和fan_out分别表示输入和输出通道数')
print('glorot_uniform,glorot_normal,he_uniform,he_normal都是比较好的初始化方法，都可以直接引用,其中后两者更适合relu,前两者更适合tanh')

from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(rescale=1./225)
train_data = generator.flow_from_directory('/Users/mingyuexu/PycharmProjects/train',target_size=(50,50),batch_size=32,class_mode='binary')
test_data = generator.flow_from_directory('/Users/mingyuexu/PycharmProjects/test',target_size=(50,50),batch_size=32,class_mode='binary')
test_label = np.hstack((np.zeros(100),np.ones(100)))

from keras.models import Sequential
from keras.layers import Dense,Activation,MaxPooling2D,Conv2D,Flatten,Dropout
from keras.initializers import HeUniform
from keras.regularizers import l1

model1 = Sequential(
    [
        Conv2D(32,(3,3),input_shape=(50,50,3),kernel_regularizer=l1(0.01)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Conv2D(32,(3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
import time
t1 = time.time()
history1 = model1.fit(train_data,epochs=10,validation_data=test_data)
t1_new = time.time()
history1_test = model1.predict_classes(test_data)
from matplotlib.pyplot import subplot
subplot(121)
plt.plot(history1.history['loss'],label='train')
plt.plot(history1.history['val_loss'],label='test')
plt.title('loss')
plt.legend()
print(train_data.class_indices)
print('fit time:',t1_new-t1)
print('accuracy1:',model1.evaluate(train_data,verbose=0)[1])
from sklearn import metrics
print('accuracy for test data:',metrics.accuracy_score(y_true=test_label,y_pred=history1_test))
print('--------------------------------------------')

model2 = Sequential(
    [
        Conv2D(32,(3,3),input_shape=(50,50,3),kernel_initializer=HeUniform(),kernel_regularizer=l1(0.01)),
        Activation('elu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Conv2D(32,(3,3)),
        Activation('elu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(128),
        Activation('elu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
t2 = time.time()
history2 = model2.fit(train_data,epochs=10,validation_data=test_data)
t2_new = time.time()
history2_test = model2.predict_classes(test_data)
subplot(122)
plt.plot(history2.history['loss'],label='train')
plt.plot(history2.history['val_loss'],label='test')
plt.title('loss')
plt.legend()
plt.show()
print('fit time:',t2_new-t2)
print('accuracy2:',model2.evaluate(train_data,verbose=0)[1])
print('accuracy for test data:',metrics.accuracy_score(y_true=test_label,y_pred=history2_test))
