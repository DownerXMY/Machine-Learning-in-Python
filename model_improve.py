import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 2

# overfit and underfit
print('The reasons for overfitting:')
print('too high dimensions')
print('too many useless variables')

print('--------------------------------------')
print('Some valid methods include PCA, AD, ... There are more:')

# One: TO apply high degree regression:
data = np.array([[46.53,48.14,50.15,51.36,52.57,54.18,56.19,58.58,61.37,63.34,65.31,66.47,68.03,69.97,71.13,71.89,73.05,74.21],
                 [2.49,2.56,2.63,2.69,2.74,2.80,2.88,2.92,2.96,2.95,2.91,2.85,2.78,2.69,2.61,2.54,2.45,2.39]]).T
# print(data)
data_X = np.array([data[:,0]]).T
data_Y = np.array([data[:,1]]).T
plt.scatter(data_X,data_Y,marker='o',color='gray')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('rate of enzymatic activity')
plt.show()

from matplotlib.pyplot import subplot
subplot(121)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(data_X,data_Y)
result = LR.predict(data_X)
from sklearn.metrics import r2_score
print('r2-score:',r2_score(y_true=data_Y,y_pred=result))
fg1 = plt.scatter(data_X,data_Y,marker='o',color='gray')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('rate of enzymatic activity')
fg2 = plt.plot(data_X,result,marker='v',color='red')
plt.scatter(np.array([[80]]),LR.predict(np.array([[80]])),color='lightgreen')
plt.xlim((40,90))
plt.legend((fg1,fg2),('original data','model prediction'))

# Clearly, it is a awful method
# Hence, we would like to improve it:
from sklearn.preprocessing import PolynomialFeatures
Poly2 = PolynomialFeatures(degree=2)
data_X_Sqrt = Poly2.fit_transform(data_X)
LR.fit(data_X_Sqrt,data_Y)
result2 = LR.predict(data_X_Sqrt)
print('r2-score for degree 2:',r2_score(y_true=data_Y,y_pred=result2))
subplot(122)
fg3 = plt.scatter(data_X,data_Y,marker='o',color='gray')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('rate of enzymatic activity')
fg4 = plt.plot(data_X,result2,marker='v',color='red')
plt.legend((fg3,fg4),('original data','d2-model prediction'))

# Moreover, we want to make a prediction about the temperature 80:
plt.scatter(np.array([[80]]),LR.predict(Poly2.transform(np.array([[80]]))),color='lightgreen')
plt.xlim((40,90))
# We can try to apply the degree5 polynomial approximation, it is obvious overfitting.
plt.show()
print('----------------------------------------')
