import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 5

# Machine Learning
# Linear Regression
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
# print(diabetes)
data_X = np.array(diabetes['data'])
data_Y = np.array(diabetes['target'])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data_X,data_Y,random_state=1,test_size=0.2)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
linear_model = LinearRegression()
linear_model.fit(X_train,Y_train)
Y_pred = linear_model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
print('classical linear regression:')
print('mse:',mean_squared_error(y_pred=Y_pred,y_true=Y_test))
print('r2-score for train set:',r2_score(y_pred=linear_model.predict(X_train),y_true=Y_train))
print('r2-score for test set:',r2_score(y_pred=Y_pred,y_true=Y_test))
plt.scatter(Y_test,Y_pred,marker='o',color='red')
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
plt.show()
print('-------------------------------')

ini = np.array([np.zeros(100)])
for Xi in X_train:
    Xi = np.array([Xi])
    Xi_T = Xi.T
    XXi = np.matmul(Xi_T,Xi)
    # print(XXi)
    XXi = XXi.reshape([1,100])
    # print(XXi)
    ini = np.vstack((ini,XXi))
data_X1 = ini[1:len(ini),:]
print('second-order linear regression:')
print(data_X1.shape)
linear_model1 = LinearRegression()
linear_model1.fit(data_X1,Y_train)

ini1 = np.array([np.zeros(100)])
for Xi in X_test:
    Xi = np.array([Xi])
    Xi_T = Xi.T
    XXi = np.matmul(Xi_T,Xi)
    # print(XXi)
    XXi = XXi.reshape([1,100])
    # print(XXi)
    ini1 = np.vstack((ini1,XXi))
data_X2 = ini1[1:len(ini1),:]
print(data_X2.shape)
Y_pred1 = linear_model1.predict(data_X2)
print('r2-score:',r2_score(y_true=Y_test,y_pred=Y_pred1))
print('mse:',mean_squared_error(y_true=Y_test,y_pred=Y_pred1))
print('-------------------------------')

# Logistic Regression
print('logistic regression:')
from sklearn.datasets import load_iris
iris = load_iris()
data2_X = np.array(iris['data'])
data2_Y = np.array(iris['target'])
# print(data2_Y)
X2_train,X2_test,Y2_train,Y2_test = train_test_split(data2_X,data2_Y,test_size=0.2,random_state=1)
print(X2_train.shape,X2_test.shape,Y2_train.shape,Y2_test.shape)
from sklearn.linear_model import LogisticRegression
logic_model = LogisticRegression()
logic_model.fit(X2_train,Y2_train)
Y2_pred = logic_model.predict(X2_test)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=Y2_test,y_pred=Y2_pred))
print('-------------------------------')
data2_X_1 = data2_X[:,[0,2]]
n = 0
for item in data2_Y:
    if item == 0:
        data2_Y[n] = 1
    else:
        data2_Y[n] = 0
    n += 1
X2_train_1,X2_test_1,Y2_train_1,Y2_test_1 = train_test_split(data2_X_1,data2_Y,test_size=0.2,random_state=1)
print(X2_train_1.shape,X2_test_1.shape,Y2_train_1.shape,Y2_test_1.shape)
logic_model1 = LogisticRegression()
logic_model1.fit(X2_train_1,Y2_train_1)
Y2_pred_1 = logic_model1.predict(X2_test_1)
print('accuracy:',metrics.accuracy_score(y_true=Y2_test_1,y_pred=Y2_pred_1))
theta_0,theta_1,theta_2 = logic_model1.intercept_[0],logic_model1.coef_[0][0],logic_model1.coef_[0][1]
x_line = np.linspace(data2_X_1[:,0].min(),data2_X_1[:,0].max(),100)
y_line = [-theta_0/theta_2-theta_1*x/theta_2 for x in x_line]

mask = logic_model1.predict(X2_train_1)==1
plt.scatter(X2_train_1[mask,0],X2_train_1[mask,1],marker='o',color='red')
plt.scatter(X2_train_1[~mask,0],X2_train_1[~mask,1],marker='x',color='blue')
mask1 = Y2_pred_1==1
plt.scatter(X2_test_1[mask1,0],X2_test_1[mask1,1],color='yellow',marker='o')
plt.scatter(X2_test_1[~mask1,0],X2_test_1[~mask1,1],color='lightgreen',marker='x')
plt.plot(x_line,y_line,linestyle='--',linewidth=5)
plt.show()
print('-------------------------------')
# another example
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
# print(breast_cancer['target'])
data2_X_2 = np.array(breast_cancer['data'])
# print(data2_X_2.shape)
data2_Y_2 = np.array(breast_cancer['target'])
X3_train,X3_test,Y3_train,Y3_test=train_test_split(data2_X_2,data2_Y_2,random_state=1,test_size=0.2)
logistic_model = LogisticRegression()
logistic_model.fit(X3_train,Y3_train)
Y3_pred = logistic_model.predict(X3_test)
print('accuracy:',metrics.accuracy_score(y_true=Y3_test,y_pred=Y3_pred))
print('--------------------------------')
# To generate a better effect:
X_Sqrt = np.array(np.zeros(900))
for item in data2_X_2:
    item = np.array([item])
    item_T = item.T
    X_sqrt = np.matmul(item_T,item)
    X_sqrt = X_sqrt.reshape([1,900])
    X_Sqrt = np.vstack((X_Sqrt,X_sqrt))
    # print(X_sqrt.shape)
X_Sqrt = X_Sqrt[1:len(X_Sqrt),:]
# print(X_Sqrt.shape)

X3_train_1,X3_test_1,Y3_train_1,Y3_test_1=train_test_split(X_Sqrt,data2_Y_2,random_state=1,test_size=0.2)
logistic_model1 = LogisticRegression()
logistic_model1.fit(X3_train_1,Y3_train_1)
Y3_pred_1 = logistic_model1.predict(X3_test_1)
print('new accuracy:',metrics.accuracy_score(y_true=Y3_test_1,y_pred=Y3_pred_1))
print('---------------------------------')