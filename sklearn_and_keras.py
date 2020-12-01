import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import keras

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris)
X = iris['data']
# print('X=',X)
Y = iris['target']
data_X = np.array(X)
data_Y = np.array(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_X,data_Y,test_size=0.2,random_state=1)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(y_true=Y_test,y_pred=Y_pred))
print(metrics.confusion_matrix(y_true=Y_test,y_pred=Y_pred))

with open('tree.txt',mode='w') as fw:
    tree.export_graphviz(clf,out_file=fw)
print('---------------------------------')

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
H = LabelBinarizer().fit_transform(data_Y)
print(H)

labels_train = LabelBinarizer().fit_transform(Y_train)
labels_test = LabelBinarizer().fit_transform(Y_test)
model = Sequential(
    [
        Dense(5,input_dim=4),
        Activation('relu'),
        Dense(3),
        Activation('sigmoid')
    ]
)
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy')
model.fit(X_train,labels_train,epochs=200,batch_size=40)
print(model.predict_classes(X_test))
print(metrics.accuracy_score(y_true=Y_test,y_pred=model.predict_classes(X_test)))
model.set_weights('./data/w')
# 下次还想使用训练出来的模型，直接
# model.load_weights('./dada/w')


