import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# 感知器
class Perception(object):
    """
    eta 学习率
    n_iter 训练(迭代)次数
    errors_ 用于记录神经元判断出错次数
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self,X,Y):
        self.w = np.zeros(1+X.shape[1])
        self.errors_ = []
        for item in range(1,self.n_iter+1,1):
            errors = 0
            for Xi,target in zip(X,Y):
                update = self.eta * (target-self.predict(Xi))
                self.w[1:] += update * Xi
                self.w[0] = update
                if update != 0:
                    errors += 1
                self.errors_.append(errors)
                pass
            pass
        pass

    def net_input(self,X):
        return np.dot(X, self.w[1:])+self.w[0]
        pass

    def predict(self,X):
        if self.net_input(X) >= 0:
            return 1
        else:
            return -1
        pass
    pass

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.keys())
data_X = iris['data']
data_Y = iris['target']
# classes = []
# c = 0
# for item in iris['target_names']:
#     h = [f'{item}--{c}']
#     c += 1
#     classes.append(h)
# print('target_names:',classes)

df = pd.DataFrame(data_X,index=pd.Series(np.arange(0,len(data_X))),columns=[0,1,2,3])
dict = {0:'setosa',1:'versicolor',2:'virginica'}
# print(dict)
df['4']=[dict[item] for item in data_Y]
print(df)

Y_whole = np.arange(1,151)
n = 0
for item in df['4']:
    # print(item)
    if item == 'setosa':
        Y_whole[n] = 1
    else:
        Y_whole[n] = -1
    n += 1
# print(Y_whole)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data_X,Y_whole,random_state=1,test_size=0.3)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

df1 = pd.DataFrame(X_train,index=pd.Series(np.arange(1,len(X_train)+1)),columns=[0,1,2,3])
df2 = pd.DataFrame(X_test,index=pd.Series(np.arange(1,len(X_test)+1)),columns=[0,1,2,3])
df1['tg'] = Y_train
# print(df1)
s = df1.loc[1:105,[0,2,'tg']].values
s_train = df1.loc[1:105,[0,2]].values
s_test = df2.loc[1:45,[0,2]].values
# print(s_test)
s1 = np.array([[0,0,0]])
s2 = np.array([[0,0,0]])
for item in s:
    if item[2] == 1:
        s1 = np.vstack((s1,np.array([item])))
    else:
        s2 = np.vstack((s2,np.array([item])))
print('-----------------------------')
s1 = s1[1:len(s1)]
s2 = s2[1:len(s2)]
print(s1,s2,sep='\n')

s3 = []
s4 = []
s5 = []
s6 = []
for item in s1:
    s3.append(item[0])
    s4.append(item[1])
for item in s2:
    s5.append(item[0])
    s6.append(item[1])
plt.scatter(s3,s4,color='red',marker='o',label='setosa')
plt.scatter(s5,s6,color='blue',marker='x',label='nonsetosa')
plt.legend(loc='upper left')
plt.show()

print('-----------------------------')
ppn = Perception(eta=0.1,n_iter=10)
ppn.fit(s_train,Y_train)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X,Y,classifier,resolution=0.02):
    marker = ['s','x','o','v']
    colors = ['red','blue','lightgreen','gray','cyan']
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    X1_min, X1_max = X[:,0].min()-1, X[:,0].max()
    X2_min, X2_max = X[:,1].min()-1, X[:,1].max()
    print(X1_min,X1_max)
    print(X2_min,X2_max)
    XX1,XX2 = np.meshgrid(np.arange(X1_min,X1_max,resolution),
                          np.arange(X2_min,X2_max,resolution))
    print(XX1.shape)
    print(XX2.shape)
    print(XX1.ravel().shape)
    print(XX2.ravel().shape)
    list1 = []
    for XI in np.array([XX1.ravel(),XX2.ravel()]).T:
        z = classifier.predict(X=XI)
        # print(z)
        list1.append(z)
    list2 = np.array(list1)
    list2 = list2.reshape(XX1.shape)
    plt.contourf(XX1,XX2,list2,alpha=0.4,cmap=cmap)
    plt.xlim(XX1.min(),XX1.max())
    plt.ylim(XX2.min(),XX2.max())

    for idx,c1 in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y==c1,0],y=X[Y==c1,1],alpha=0.8,cmap=cmap(idx),marker=marker[idx],label=c1)

plot_decision_regions(X=s_test,Y=Y_test,classifier=ppn,resolution=0.02)
plt.legend(loc='upper left')
plt.show()

# 适应性神经元
class AdalineGD(object):
    def __init__(self,eta,n_iter):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self,X,Y):
        self.w = np.zeros(1+X.shape[1])
        self.cost = []
        for item in range(self.n_iter):
            output = self.net_input(X)
            errors = Y - output
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]

    def activate(self,X):
        return self.net_input(X)

    def predict(self,X):
        if self.activate(X) >= 0:
            return 1
        else:
            return -1
        pass
    pass

iris1 = load_iris()
data_set = np.array(iris1['data'])
# print(data_set)
input_set = data_set[:,[0,2]]
# print(input_set)
result = []
for item in iris1['target']:
    if item == 0:
        result.append(1)
    else:
        result.append(-1)
result_set = np.array(result)
# print(result_set)

input_train,input_test,result_train,result_test = train_test_split(input_set,result_set,random_state=1)
print(input_train.shape,input_test.shape,result_train.shape,result_test.shape)
# print(input_train)
# print(result_train)
whole_set = np.hstack((input_train,np.array([result_train]).T))
print(whole_set)
list1 = []
list2 = []
for item in whole_set:
    if item[2] == 1:
        list1.append(item)
    else:
        list2.append(item)
plt.scatter([item[0] for item in list1],[item[1] for item in list1],color='red',marker='o',label='class1')
plt.scatter([item[0] for item in list2],[item[1] for item in list2],color='blue',marker='x',label='class2')
plt.show()

ada = AdalineGD(eta=0.0001,n_iter=50)
ada.fit(X=input_train,Y=result_train)

def apply_AdalineGD(X,Y,Z,classifier,resolution=0.02):
    marker = ['x','o','s','v']
    color_map = ['red','blue','green','gray']
    cmap = ListedColormap(color_map[:len(np.unique(Y))])
    X1_min,X1_max = X[:,0].min()-1, X[:,0].max()
    X2_min,X2_max = X[:,1].min()-1, X[:,1].max()
    XX1,XX2 = np.meshgrid(np.arange(X1_min,X1_max,resolution),
                          np.arange(X2_min,X2_max,resolution))

    list4 = []
    for X_i in np.array([XX1.ravel(),XX2.ravel()]).T:
        rel = classifier.predict(X=X_i)
        list4.append(rel)

    list5 = np.array(list4)
    list5 = list5.reshape(XX1.shape)
    # print(list5)
    plt.contourf(XX1,XX2,list5,alpha=0.4,cmap=cmap)
    plt.xlim(XX1.min(), XX1.max())
    plt.ylim(XX2.min(), XX2.max())

    for idx,c1 in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y==c1,0],y=X[Y==c1,1],alpha=0.8,cmap=cmap(idx),marker=marker[idx],label=c1)

apply_AdalineGD(input_set,result_set,input_test,classifier=ada,resolution=0.02)
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost)+1),ada.cost,marker='x')
plt.xlabel('epochs')
plt.ylabel('sum_squared_error')
plt.show()