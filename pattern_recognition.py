import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# from keras.datasets import mnist
# (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
S = np.load('mnist.npz')
X_train,Y_train,X_test,Y_test = S['x_train'],S['y_train'],S['x_test'],S['y_test']
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
from matplotlib.pyplot import subplot
for item in range(1,10,1):
    subplot(3,3,item)
    img = X_train[item-1]
    plt.imshow(img)
    plt.title(Y_train[item-1])
plt.show()

print('---------------------------------')
feature_size = X_train[0].shape[0] * X_train[0].shape[1]
X_train_format = X_train.reshape(X_train.shape[0],feature_size)
X_test_format = X_test.reshape(X_test.shape[0],feature_size)
print(X_train_format.shape,X_test_format.shape)
X_train_normal = X_train_format/255
X_test_normal = X_test_format/255
from keras.utils import to_categorical
Y_train_normal = to_categorical(Y_train,num_classes=10)
Y_test_normal = to_categorical(Y_test,num_classes=10)
# print(Y_train_normal[0])

from keras.models import Sequential
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

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train_normal,Y_train_normal,epochs=10)
result = model.predict_classes(X_test_normal)
score = model.evaluate(X_test_normal,Y_test_normal,verbose=0)
print('loss=',score[0])
print('accuracy=',score[1])

for item in range(1,10,1):
    subplot(3,3,item)
    img1 = X_test[item-1]
    plt.imshow(img1)
    plt.title(result[item-1])
plt.show()
print('--------------------------------------')