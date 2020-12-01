import math
import sys
import numpy as np

print(sys.path)
if '/Users/xmy/PycharmProjects/demo' not in sys.path:
    sys.path.append('/Users/xmy/PycharmProjects/demo')
f = open('/Users/mingyuexu/PycharmProjects/demo/learning/media_0731.csv', mode ='r')
data_X = []
data_Y = []
data_Z = []
data = []
n = 0
for line in f.readlines():
    data1 = []
    n = n + 1
    if n == 1:
        continue
    else:
        # print(line)
        data1 = str.split(line, sep = ',')
        for i in range(1,len(data1)+1,1):
            if i == 1:
                data1[i-1] = str(data1[i-1])
            elif i == 3 or i == 6:
                data1[i-1] = float(math.log(float(data1[i-1]),2))
            else:
                data1[i-1] = float(data1[i-1])
        # print(data1, len(data1))
        data.append(data1)
f.close()

for j in range(1,len(data)+1,1):
    k = data[j-1]
    j_1 = k[1:10]
    # print(j_1)
    j_2 = k[-1:]
    # print(j_2)
    j_3 = k[10:-1]
    data_X.append(j_1)
    data_Y.append(j_2)
    data_Z.append(j_3)
X = np.array(data_X)
Y = np.array(data_Y)
Z = np.array(data_Z)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
regr = MLPRegressor(random_state=0, learning_rate_init=1e-5, solver='adam', max_iter=1000000, hidden_layer_sizes=[7,7])
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)
print('Y_test=',Y_test)
print('Y_pred=',Y_pred)
print('--------------------------------')
s = np.arange(1,15)
import matplotlib.pyplot as plt
plt.scatter(s, Y_test)
plt.scatter(s, Y_pred)
plt.show()

print('Neural Networks:')
print('train data score:', regr.score(X_train, Y_train))
print('test data score:', regr.score(X_test, Y_test))

# Z = np.array(data_Z)
# Y = np.array(data_Y)
# Z -= np.mean(Z,axis=0)
# Z /= np.std(Z,axis=0)
# from sklearn.model_selection import  train_test_split
# Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size = 0.2)
#
# from sklearn.linear_model import LinearRegression
# regr1 = LinearRegression().fit(Z_train,Y_train)
# print(regr1.score(Z_train,Y_train))
# print(regr1.score(Z_test,Y_test))
print('-----------------------------------------------')

from keras.models import  Sequential
from keras.layers import  Dense,Activation
from keras.optimizers import SGD
from keras.constraints import unit_norm

model = Sequential(
    [
        Dense(7,input_dim=9),
        Activation('relu'),
        Dense(7),
        Activation('relu'),
        Dense(1)
    ]
)
sgd = SGD(lr=0.001,decay=1e-5,momentum=0.5,nesterov=True)
model.compile(optimizer=sgd,loss="mean_absolute_error")
model.fit(X_train,Y_train,epochs=100,batch_size=40)
print(model.predict_classes(X_test))
plt.scatter(s,Y_test)
plt.scatter(s,model.predict_classes(X_test))
plt.show()




