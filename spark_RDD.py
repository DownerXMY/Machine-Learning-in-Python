# Resilient Distributed Dataset

from pyspark.conf import SparkConf
from pyspark.context import SparkContext

conf = SparkConf().setMaster('local[2]').setAppName('spark_demo1')
sc = SparkContext(conf=conf)
data = [1,2,3,4,5]
distData = sc.parallelize(data,5)
print(distData.collect())
distData.foreach(lambda x:print(x))
sum = distData.reduce(lambda a,b:a+b)
print(sum)
print(distData.map(lambda x:x+1).collect())
print(distData.filter(lambda x:x>3).collect())
print(distData.groupBy(lambda x:x%2==0).collect())
data1 = ['hello spark','hello world','much love']
distData1 = sc.parallelize(data1,3)
print(distData1.map(lambda x:[word for word in x.split(' ')]).collect())
print(distData1.flatMap(lambda x:x.split(' ')).collect())
print(distData1.flatMap(lambda x:x.split(' ')).map(lambda x:(x,1)).groupByKey().map(lambda x:{x[0]:list(x[1])}).collect())
data2 = [2,2,4,5,3,4,2]
distData2 = sc.parallelize(data2)
print(distData2.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).sortBy(lambda x:x[1],ascending=False).collect())
data3 = [3,4,5,6,7]
distData3 = sc.parallelize(data3)
print(distData.map(lambda x:(x,'x')).join(distData3.map(lambda x:(x,'y'))).collect())
print(distData.map(lambda x:(x,'x')).leftOuterJoin(distData3.map(lambda x:(x,'y'))).collect())

import numpy as np
data4 = np.arange(1,10,1)
rdd4 = sc.parallelize(data4)
print(rdd4.count(),rdd4.take(4),rdd4.max(),rdd4.mean(),sep='  ')
data5 = np.arange(1,11,1).reshape(2,5)
rdd5 = sc.parallelize(data5)
print(rdd5.collect())
data6 = np.arange(2,11,1)
rdd6 = sc.parallelize(data6)
print(rdd4.zip(rdd6).collect())
sc.stop()

import pandas as pd
df = pd.read_csv('/Users/mingyuexu/PycharmProjects/demo/金庸-射雕英雄传txt精校版.txt',names=['txt'],encoding='GBK')
print(len(df))

for num in range(0,len(df),1):
    if len(df.txt[num]) <= 2:
        df = df.drop(index=num)
chapters = []
for item in range(1,len(df)+1,1):
    if df.txt[df.index[item-1]][0] == '第':
        validation = True
        n = 0
        while validation:
            n += 1
            validation = item-1+n <= len(df)-1 and df.txt[df.index[item-1+n]][0] != '第'
        chapter = df.loc[df.index[item-1]:df.index[item-2+n],:]
        chapters.append(chapter)
print(len(chapters))

m = -1
for sentence in chapters[39].txt:
    m += 1
    if sentence[0:2] == '附录':
        break
chapters[39] = chapters[39].loc[chapters[39].index[0]:chapters[39].index[m-2],:]
# print(chapters[39])

text_chapters = []
for chapter_df in chapters:
    ori = ''
    for num in range(0,len(chapter_df),1):
        if num == 0:
            ori = ori + chapter_df.txt[chapter_df.index[num]]
        else:
            ori = ori + chapter_df.txt[chapter_df.index[num]][2:]
    text_chapters.append(ori)
print(len(text_chapters),text_chapters[0])

import time
t1 = time.time()
conf1 = SparkConf().setMaster('local[2]').setAppName('spark_demo2')
sc1 = SparkContext(conf=conf1)
rdd_chapters = sc1.parallelize(text_chapters,40)
def My_func(x):
    dict = {}
    for pair in x:
        if pair[0] not in dict.keys():
            dict[pair[0]] = 1
        else:
            dict[pair[0]] += 1
    return dict
after_dis = rdd_chapters.map(lambda x:[(x[item],1) for item in range(0,len(x),1)]).map(My_func).collect()

whole_dict = {}
for item in after_dis:
    for word in item.keys():
        if word not in whole_dict.keys():
            whole_dict[word] = item[word]
        else:
            whole_dict[word] = whole_dict[word] + item[word]
vocabulary = sorted(whole_dict.items(),key=lambda x:x[1],reverse=True)
t2 = time.time()
sc1.stop()
print('time in pyspark:',t2-t1)
print(vocabulary)

t3 = time.time()
dict1 = {}
for article in text_chapters:
    for word in article:
        if word not in dict1.keys():
            dict1[word] = 1
        else:
            dict1[word] += 1
vocabulary1 = sorted(dict1.items(),key=lambda x:x[1],reverse=True)
t4 = time.time()
print('time in banel:',t4-t3)
print(vocabulary1)
print('--------------------------------------')
# 分布式处理速度非常快,尤其是数据量巨大的时候！
