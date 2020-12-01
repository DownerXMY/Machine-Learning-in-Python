import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Learning Scipy
# integral
from scipy.integrate import quad, dblquad, nquad
print(quad(lambda x: np.exp(-x), 0, np.inf))
# 返回的第一个是值，第二个是误差范围
print(dblquad(lambda x, y: np.exp(-x * y) / x ** 3, 0, np.inf, lambda x: 1, lambda x: np.inf))
print(dblquad(lambda x, y: np.exp(-x * y) / x ** 3, 0, np.inf, 1, np.inf))
# 第一个表达式告诉我们，y的定义域可以表示成x的函数
def mul(x, y):
    return x * y
def bound_y():
    return [0, 0.5]
def bound_x(y):
    return [0, 1 - 2 * y]
print(nquad(mul, [bound_x, bound_y]))
print('---------------------------------------------')

# optimizer
from scipy.optimize import minimize
def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
x0 = np.array([1.3, 0.7, 0.8, 0.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print('ROSE MINI:', res)
print(res.x)
# 在一定的subject下求最小值
def func(x):
    return 2 * x[0] * x[1] + 2 * x[0] - x[0] ** 2 - 2 * x[1] ** 2
def func1(x):
    prime_x0 = -2 * x[0] + 2 * x[1] + 2
    prime_x1 = 2 * x[0] - 4 * x[1]
    return np.array([prime_x0, prime_x1])
cons = ({'type': 'eq', 'fun': lambda x: np.array([x[0] ** 3 - x[1]]), 'jac': lambda x: np.array([3 * (x[0] ** 2), -1])},
        {'type': 'ineq', 'fun': lambda x: np.array([x[1] - 1]), 'jac': lambda x: np.array([0, 1])})
# 这时候我们的限制条件就是：x[0]**3-x[1]=0和x[1]-1!=0，事实上后面可以完全不用再写雅克比jac，写出来是为了提升计算速度
res1 = minimize(func,[-1.0,1.0],jac=func1,constraints=cons,method='SLSQP',options={'disp': True})
print('func MINI:', res1)
from scipy.optimize import root
def func2(x):
    return x + 2*np.cos(x)
res2 = root(func2,0.1)
print('res2:',res2)
print('---------------------------------------------')

# interpolation
from matplotlib.pylab import *
from scipy.interpolate import interp1d
x = np.linspace(0,1,10)
y = np.sin(2*np.pi*x)
li = interp1d(x,y,kind='cubic')
x_new = np.linspace(0,1,50)
y_new = li(x_new)
figure()
plot(x,y,'r')
plot(x_new,y_new,'k')
show()
print(y_new)
print('---------------------------------------------')

# Linear
from scipy import linalg as lg
m = np.array([[1,2],[3,4]])
print('DET:',lg.det(m))
print('INV:',lg.inv(m))
s = np.array([6,14])
print('Sol:',lg.solve(m,s))
print('Eigenvectors:',lg.eig(m))
print('LU分解:',lg.lu(m))
print('QR分解:',lg.qr(m))
print('SVD:',lg.svd(m))
print('舒尔分解:',lg.schur(m))
print('---------------------------------------------')

# Learning Pandas
# series and dataframe
import pandas as pd
s = pd.Series([i**2 for i in range(1,11)])
# print(s,type(s))
d = pd.date_range('20200808',periods=7)
df = pd.DataFrame(np.random.randn(7,5),index=d,columns=list('ABCDE'))
print(df,type(df))
print('---------------------------------------------')
# print(df is not iterable)

# Basic matipulations
print(df.head(3))
print(df.tail(3))
print(df.index)
print(df.values,type(df.values))
print(df.T)
print(df.sort_index(axis=1,ascending=False))
print(df.describe())
print(df['C'],type(df['C']))
print(df['20200808':'20200810'])
print(df.at[d[1],'B'])
print(df.iloc[3,4])
print(df[df.B>0])
print(df[df.B>0][df.C>0])
print('---------------------------------------------')
s1 = pd.Series([item for item in range(10,17)],index=pd.date_range('20200808',periods=7))
df['F'] = s1
print(df)
df.at[d[0],'A']=0
print(df)
df1 = df.copy()
df1[df1<0] = -df1
print(df1)
print('---------------------------------------------')

# Missing data
df2 = df.reindex(index=d[:4],columns=list('ABCD')+['G'])
df2.loc[d[0]:d[1],'G']=1
print(df2)
# 处理缺失值的两种方法分别是丢弃和填充
print(df2.dropna())
print(df2.fillna(value=1))
print('---------------------------------------------')

# Statistics
print(df.mean())
print(df.var())
s2 = pd.Series([item for item in range(1,8)],index=d,dtype=np.float)
print(s2)
print(s2.shift(2))
print(s2.diff())
s3 = pd.Series([item for item in range(1,8)],index=d,dtype=np.float)
s3.loc[d[2]:d[4]]=7
print(s3)
print(s3.value_counts())
print(df)
print(df.apply(np.cumsum)) # 每一行都是前面行的累加
print(df.apply(lambda x:x.max()-x.min())) # 自定义算法
print('---------------------------------------------')

# 表格拼接
print(df[:3])
print(df[-3:])
pieces = [df[:3],df[-3:]]
print(pd.concat(pieces))
#
df_left = pd.DataFrame({'key':['x','y'],'valye':[1,2]})
df_right = pd.DataFrame({'key':['x','z'],'valye':[3,4]})
# 当然也可以继续用我们之前的定义方法
# df_left1 = pd.DataFrame(np.array([['x','y'],[1,2]]).T,index=pd.Series([0,1]),columns=['key','value'])
# df_right1 = pd.DataFrame(np.array([['x','z'],[3,4]]).T,index=pd.Series([0,1]),columns=['key','value'])
# print(df_left1,df_right1,sep='\n')
print(df_left,df_right,sep='\n')
print(pd.merge(df_left,df_right,on='key',how='left'))
print(pd.merge(df_left,df_right,on='key',how='right'))
print(pd.merge(df_left,df_right,on='key',how='inner'))
print(pd.merge(df_left,df_right,on='key',how='outer'))
print('---------------------------------------------')
df4 = pd.DataFrame(np.array([
    ['a','b','c','b'],
    [0,1,2,3]
]).T,index=pd.Series([0,1,2,3]),columns=list('AB'))
print(df4)
print(df4.groupby('A').sum())
# 注意上下两种的结果是不一样的，以下是具体原因
# print(type(df4.loc[pd.Series([0,1,2,3])[1],'B']))
df5 = pd.DataFrame({'A':['a','b','c','b'],'B':[0,1,2,3]})
print(df5)
print(df5.groupby('A').sum())
print('---------------------------------------------')
# 透视功能
df6 = pd.DataFrame({
    'A':['one','two','three','four']*6,
    'B':['a','b','c']*8,
    'C':['tets1','test1','test1','test2','test2','test2']*4,
    'D':np.random.randn(24),
    'E':np.random.randn(24),
    'F':[datetime.datetime(2020,i,1) for i in range(1,13)]+[datetime.datetime(2020,i,15) for i in range(1,13)]
})
print(pd.pivot_table(df6,values='D',index=['A','B'],columns=['C']))
print('---------------------------------------------')
# 绘图
t_exam = pd.date_range('20200809',periods=7,freq='S')
print(t_exam)
ts = pd.Series(np.random.randn(1000),index=pd.date_range('20200809',periods=1000,freq='S'))
ts = ts.cumsum()
from pylab import *
ts.plot()
show()
print('---------------------------------------------')

