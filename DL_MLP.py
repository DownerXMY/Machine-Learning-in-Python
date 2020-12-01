import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
data_X = np.array(bc['data'])
data_Y = np.array(bc['target'])
import pandas as pd
H = pd.Series(data_Y)
print(H.value_counts())
print(data_X.shape,data_Y.shape)
print('-------------------------------------')
print('sklearn-NN:')
from sklearn.neural_network import MLPClassifier
MLPC = MLPClassifier(hidden_layer_sizes=(30,20),activation='relu',solver='adam',learning_rate_init=1e-4,momentum=0.9,max_iter=10000)
MLPC.fit(data_X,data_Y)
res = MLPC.predict(data_X)
# print(res)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=res))
print('-------------------------------------')

print('Keras:')
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
model = Sequential(
    [
        Dense(20,input_dim=30),
        Activation('relu'),
        Dense(20),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ]
)
sgd = SGD(learning_rate=0.001,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='binary_crossentropy')
model.fit(data_X,data_Y,batch_size=100,epochs=1000,verbose=0)
result = model.predict_classes(data_X)
# print(result)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result))
print('-------------------------------------')

print('Multi-classify problem:')
print('sigmoid:')
from sklearn.datasets import load_iris
iris = load_iris()
data_X1 = iris['data']
data_Y1 = iris['target']
from sklearn.preprocessing import LabelBinarizer
data_Y1_up = LabelBinarizer().fit_transform(data_Y1)

model1 = Sequential(
    [
        Dense(4,input_dim=4),
        Activation('relu'),
        Dense(3),
        Activation('sigmoid')
    ]
)
model1.compile(optimizer=sgd,loss='binary_crossentropy')
model1.fit(data_X1,data_Y1_up,batch_size=50,epochs=500,verbose=0)
result_2 = model1.predict_classes(data_X1)
# print(result_2)
print('accuracy:',metrics.accuracy_score(y_true=data_Y1,y_pred=result_2))

print('-------------------------------------')
print('softmax:')
model2 = Sequential(
    [
        Dense(4,input_dim=4),
        Activation('relu'),
        Dense(4),
        Activation('relu'),
        Dense(3),
        Activation('softmax')
    ]
)
model2.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
import keras
data_Y1_up2 = keras.utils.to_categorical(data_Y1, num_classes=3)
model2.fit(data_X1,data_Y1_up2,batch_size=50,epochs=500,verbose=0)
result2 = model2.predict_classes(data_X1)
score = model2.evaluate(data_X1,data_Y1_up2,verbose=0)
print('loss:',score[0])
print('accuracy:',score[1])

