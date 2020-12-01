import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

print('including Tree, Anomaly Detection, PCA, ...')
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
data_X = np.array(bc['data'])
data_Y = np.array(bc['target'])
print(data_X.shape,data_Y.shape)
# print(bc['feature_names'])
print('-------------------------------------')

# Decision Tree:
print('tree:')
from sklearn.tree import DecisionTreeClassifier, plot_tree
TR = DecisionTreeClassifier(criterion='entropy')
TR.fit(data_X,data_Y)
result1 = TR.predict(data_X)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=data_Y,y_pred=result1))

# visualise the tree:
plt.figure(figsize=(10,5))
plot_tree(TR,filled=True,feature_names=bc['feature_names'],class_names=['yes','no'])
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data_X,data_Y,test_size=0.3,random_state=1)
TR1 = DecisionTreeClassifier(criterion='entropy')
TR1.fit(X_train,Y_train)
result1_1 = TR1.predict(X_test)
print('accuracy of test-data:',metrics.accuracy_score(y_true=Y_test,y_pred=result1_1))

# Some tecs to raise the accuracy:
# Add the min_samples_leaf coefficient:
TR2 = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=10)
TR2.fit(X_train,Y_train)
result1_2 = TR2.predict(X_test)
print('accuracy after adjustment:',metrics.accuracy_score(y_true=Y_test,y_pred=result1_2))
print('--------------------------------------')

print('Anomaly Detection:')
from sklearn.datasets import load_iris
iris = load_iris()
# print(iris['feature_names'])
data_X1 = np.array(iris['data'])
data_Y1 = np.array(iris['target'])
# print(data_X1.shape,data_Y1.shape)
whole_data = np.hstack((data_X1,np.array([iris['target']]).T))
label1 = []
for item in whole_data:
    if item[4] == 0:
        label1.append(item)
label1 = np.array(label1)
label1 = label1[:,[0,2]]
plt.scatter(label1[:,0],label1[:,1],color='blue',marker='o')
plt.xlim((0,10))
plt.ylim((0,5))
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()
# Add a anomaly data:
label1_for_AD = np.vstack((label1,np.array([[7,1.5]])))
print(label1_for_AD.shape)

# Visualise the distribution of each variable:
from matplotlib.pyplot import subplot
subplot(121)
plt.hist(label1_for_AD[:,0],bins=100)
plt.title('sepal length distribution')
subplot(122)
plt.hist(label1_for_AD[:,1],bins=100)
plt.title('petal length distribution')
plt.show()

# calculate information of the variables:
x1_mean = np.mean(label1_for_AD[:,0])
x1_std = np.std(label1_for_AD[:,0])
print(f'mean of sepal length={x1_mean}, std of sepal length={x1_std}')
x2_mean = np.mean(label1_for_AD[:,1])
x2_std = np.std(label1_for_AD[:,1])
print(f'mean of petal length={x2_mean}, std of petal length={x2_std}')

# calculate the Gaussian distribution
from scipy.stats import norm
x1_normal = norm.pdf(np.linspace(0,10,100),x1_mean,x1_std)
x2_normal = norm.pdf(np.linspace(0,5,100),x2_mean,x2_std)
subplot(121)
plt.plot(np.linspace(0,10,100),x1_normal)
plt.title('normal--sepal length')
subplot(122)
plt.plot(np.linspace(0,5,100),x2_normal)
plt.title('normal--petal length')
plt.show()

# Find the anomaly:
from sklearn.covariance import EllipticEnvelope
AD = EllipticEnvelope()
AD.fit(label1_for_AD)
result2 = AD.predict(label1_for_AD)
# print(result2)
result2_2 = np.array([result2]).T
label1_for_AD = np.hstack((label1_for_AD,result2_2))
# print(label1_for_AD.shape)

# Visualise the result:
subplot(121)
plt.scatter(label1_for_AD[:,0],label1_for_AD[:,1],color='blue',marker='o')
plt.xlim((0,10))
plt.ylim((0,5))
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('original data')
subplot(122)
plt.scatter(label1_for_AD[label1_for_AD[:,2]==1,0],label1_for_AD[label1_for_AD[:,2]==1,1],color='blue',marker='o')
plt.scatter(label1_for_AD[label1_for_AD[:,2]==-1,0],label1_for_AD[label1_for_AD[:,2]==-1,1],color='red',marker='x',linewidths=2)
plt.xlim((0,10))
plt.ylim((0,5))
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('Anomaly Detection')
plt.show()
num = 0
for item in label1_for_AD:
    if item[2] == -1:
        num += 1
print('the number of anomalies:',num)
print('---------------------------------------')

# PCA
print('principle component analysis:')
bc1 = load_breast_cancer()
data_X2 = bc1['data']
data_Y2 = bc1['target']
print(data_X2.shape,data_Y2.shape)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(data_X2,data_Y2)
result3 = clf.predict(data_X2)
print('accuracy:',metrics.accuracy_score(y_true=data_Y2,y_pred=result3))
# To get the data standard(mean=0,std=1):
from sklearn.preprocessing import StandardScaler
X2_norm = StandardScaler().fit_transform(data_X2)
# To apply the PCA analysis:
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X2_pca = pca.fit_transform(X2_norm)
print(pca.explained_variance_ratio_)
# We can see that the principle components can be the first 10 terms
pca1 = PCA(n_components=10)
X2_pca1 = pca1.fit_transform(X2_norm)
clf1 = KNeighborsClassifier(n_neighbors=2)
clf1.fit(X2_pca1,data_Y2)
result3_2 = clf1.predict(X2_pca1)
print('accuracy after PCA:',metrics.accuracy_score(y_true=data_Y2,y_pred=result3_2))

# The most important utilization for PCA is visualising:
data_X3 = np.array(iris['data'])
data_Y3 = np.array(iris['target'])
print(data_X3.shape,data_Y3.shape)
pca_vis = PCA(n_components=4)
data_X3_update = pca_vis.fit_transform(data_X3)
print(pca_vis.explained_variance_ratio_)
pca_vis2 = PCA(n_components=3)
data_X3_reupdate = pca_vis2.fit_transform(data_X3)
from sklearn.linear_model import LogisticRegression
logic_model = LogisticRegression()
logic_model.fit(data_X3_reupdate,data_Y3)
result4 = logic_model.predict(data_X3_reupdate)
print('accuracy:',metrics.accuracy_score(y_true=data_Y3,y_pred=result4))
from mpl_toolkits.mplot3d import Axes3D
ax = plt.gca(projection='3d')
ax.scatter(data_X3_reupdate[result4==0,0],data_X3_reupdate[result4==0,1],data_X3_reupdate[result4==0,2],color='red')
ax.scatter(data_X3_reupdate[result4==1,0],data_X3_reupdate[result4==1,1],data_X3_reupdate[result4==1,2],color='blue')
ax.scatter(data_X3_reupdate[result4==2,0],data_X3_reupdate[result4==2,1],data_X3_reupdate[result4==2,2],color='lightgreen')
plt.show()
