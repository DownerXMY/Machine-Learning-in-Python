import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
print (mpl.matplotlib_fname())
from matplotlib.font_manager import _rebuild
_rebuild()
# from matplotlib import font_manager
# a = sorted([f.name for f in font_manager.fontManager.ttflist])
# for i in a:
#     print(i)
#
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.titlecolor'] = 'r'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['lines.color'] = 'red'
plt.rcParams['lines.linestyle'] = '--'
x = [1,2]
y = [-3,4]
plt.title('中文标题')
plt.plot(x,y)
plt.show()

# 直方图
height = [165,158,182,170,161,155,173,176,181,190,172,163,170]
bins = range(150,200,5)
plt.hist(height, bins=bins)
plt.show()

# 条形图
classes = ['class1','class2','class3']
scores = [70,60,80]
plt.bar(classes, scores)
plt.show()

# 折线图
height = [165,158,182,170,161,155,173,176,181,190,172,163,170,178,183]
year = range(2005,2020)
plt.plot(year, height)
plt.show()

# 饼图
labels = ['房贷','饮食','出行','教育']
money = [8000,2000,2000,3000]
plt.pie(money, labels=labels, autopct='%1.1f%%')
plt.show()

# 散点图
f = open('/Users/mingyuexu/PycharmProjects/demo/learning/media_0731.csv', mode ='r')
R = []
R1 = []
X1 = []
TS = []
X = []
Y = []
n = 0
for line in f.readlines():
    R = str.split(line, sep=',')
    n = n + 1
    if n != 1:
        R1 = R[1:]
        # print(R1)
        X1.append(R1)
TS = np.array(X1)
TS = np.sort(TS,axis=0)
# print(TS)
X = TS[:,0]
Y = TS[:,-1]
# print(X)
f.close()

plt.scatter(X,Y,color='r')
plt.title('发文数与PDI的关系')
plt.xlabel('发文数')
plt.ylabel('PDI')
plt.show()

# 分图
fig = plt.figure()
fig.add_subplot(3,3,1)
n = 128
XX = np.random.normal(0,1,n)
YY = np.random.normal(0,1,n)
ARC = np.arctan2(YY,XX)
plt.scatter(XX,YY,s=75,c=ARC,alpha=0.5)
plt.xlim(-1.5,1.5), plt.xticks([])
plt.ylim(-1.5,1.5), plt.yticks([])
plt.axis()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

