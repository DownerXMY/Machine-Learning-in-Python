import numpy as np
a = np.array([1,2,3])
print(a)
print(a.dtype)
b = np.array([1,3,5,7], dtype=np.int32)
print(b, b.dtype)
c = np.arange(start=0,stop=10,step=2)
d = np.linspace(start=0,stop=10,num=5,endpoint=False)
e = np.logspace(start=1,stop=5,base=2,num=5)
print(c,d,e)
print('-----------------------------')
a1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
a2 = a1.T
print(a1,a2,sep='\n')
print('-----------------------------')
a3 = np.ones([2,3],dtype=np.int)
a4 = np.zeros([3,2],dtype=np.int)
a5 = np.full([3,4],5)
a6 = np.identity(2,dtype=np.int)
print(a3,a4,a5,a6,sep='\n')
print('-----------------------------')
b1 = np.array([[1,3,5],
              [2,4,6],
              [7,8,9]])
print(b1[1][2])
print(b1[2,2])
# 先行后列
print(b1[1,:])
print(b1[:,1])
for item in b1[2,:]:
    print(item)
print('-----------------------------')
# 布尔索引
b3 = np.array([1,2,3,4,5,6],dtype=np.int)
b4 = np.array([1,1,1,1,1,1],dtype=np.bool)
b5 = np.array([False,True,False,True,False,True])
for item in b3:
    if item % 2 == 0:
        b4[item-1] = True
    else:
        b4[item-1] = False
print(b4)
print(b3[b4])
print(b3[b5])
print('-----------------------------')
c1 = np.array([[1,2,3],
               [4,5,6],
               [7,8,9]])
c2 = np.array([[1,2,7],
               [3,4,8],
               [5,6,9]])
c3 = np.concatenate((c1,c2))
c4 = np.concatenate((c1,c2),axis=1)
c5 = np.vstack((c1,c2))
c6 = np.hstack((c1,c2))
print(c3,c4,c5,c6,sep='\n')
print('-----------------------------')
d1 = np.array([[1,2,3,4],
               [5,6,7,8],
               [9,10,11,12]])
d2 = np.split(d1,np.array([0,1]))
d3 = np.split(d1,3)
d4 = np.split(d1,np.array([1,3]),axis=1)
d5 = np.vsplit(d1,3)
d6 = np.hsplit(d1,2)
print(d2,d3,d4,d5,d6,sep='\n')
print('-----------------------------')
e1 = np.array([1,2,3,4,5])
e2 = e1.T
print(e1 + 2)
print(e1 ** 2)
print(e1 * e1)
print(e1 * e2)
print(np.matmul(e1,e2))
e3 = np.array([[1,2,3],
               [4,5,6]])
print(np.matmul(e3,e3.T))
print('-----------------------------')
f1 = np.random.rand(10)
f2 = np.random.randint(1,100,10)
f3 = np.random.normal(3,4,10)
f4 = np.random.randn(10)
print(f1,f2,f3,f4,sep='\n')
print('-----------------------------')
g = np.random.randint(1,100,10)
g1 = np.sort(g)
g2 = np.argsort(g,axis=-1)
print(g,g1,g2,sep='\n')
print('-----------------------------')
h = np.random.randn(10)
h[9] = np.nan
h1 = np.sum(h)
h2 = np.nansum(h)
h3 = h.max()
h4 = np.amax(h)
h5 = np.nanmax(h)
h6 = np.mean(h)
h7 = np.nanmean(h)
h8 = np.median(h)
h9 = np.average(h)
h10 = np.nanmedian(h)
print(h,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,sep='\n')
print('-----------------------------')
i1 = np.array([[1,2,3],
               [4,5,6],
               [7,8,9]])
np.save('test',i1)
i2 = np.array([[1,2],
              [3,4]])
np.savez('test1',array1=i1,array2=i2)
print('-----------------------------')
print(np.load('test.npy'))
t1 = np.load('test1.npz')
print(t1['array2'])
print('-----------------------------')
list = np.arange(1,11).reshape([2,5])
print(list)
print(np.exp(list))
print(np.log(list))
print(np.exp2(list))
print('-----------------------------')
list1 = np.arange(1,6)
print(list,list1,sep='\n')
print(np.dot(list,list1))
print('-----------------------------')
from numpy.linalg import *
j1 = np.arange(1,5).reshape([2,2])
print('inverse = ',inv(j1),sep='\n')
print(j1.transpose())
print(j1.T)
print(det(j1))
answer = np.array([[5],[7]])
print(solve(j1, answer))
print('-----------------------------')


