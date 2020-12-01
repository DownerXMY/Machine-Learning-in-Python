import numpy as np
import nltk
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# run the following code if you want to download newers from ntlk
# nltk.download()

from nltk.corpus import brown
print(brown.categories())
print(len(brown.sents()),len(brown.words()))

df = pd.read_table('/Users/mingyuexu/PycharmProjects/demo/金庸-射雕英雄传txt精校版.txt',names=['txt'],encoding='GBK')
print(len(df))

for num in range(0,len(df),1):
    if len(df.txt[num]) <= 2:
        df = df.drop(index=num)
# print(df)
chapter = []
ls = df.index
for item in range(0,len(df),1):
    if df.txt[ls[item]][0] == '第' and len(df.loc[ls[item],'txt']) <= 12:
        m = 0
        judge = True
        while judge:
            m += 1
            judge = df.loc[ls[item+m],'txt'][0] != '第' and ls[item+m] <= len(ls)-1
        chapter.append(df.loc[ls[item]:ls[item+m-1],:])
print('How many chapters:',len(chapter))

ls1 = chapter[39].index
for item in range(0,len(chapter[39]),1):
    if chapter[39].loc[ls1[item],'txt'][0:2] == '附录':
        chapter[39] = chapter[39].loc[ls1[0]:ls1[item - 1]]
        break
# print(chapter[39])

chapter_cor = []
for item in chapter:
    s = 0
    chapter_up = ''
    for sentence in item.txt:
        s += 1
        if s == 1:
            chapter_up = chapter_up + sentence
        else:
            chapter_up = chapter_up + sentence[2:]
    chapter_cor.append(chapter_up)
print(chapter_cor[0])
print('----------------------------------')

import jieba.posseg as psg
stoptable = pd.read_table('/Users/mingyuexu/PycharmProjects/demo/停用词.txt',names=['w'],encoding='utf-8')
stoplist = []
for item in range(1,len(stoptable)+1,1):
    stoplist.append(stoptable.w[item-1])
wordlist = []
for chapters in chapter_cor:
    wordlist_individual = [pair.word for pair in psg.lcut(chapters) if pair.word not in stoplist]
    wordlist.append(wordlist_individual)
# pair = (word, attribute)
print('after cut:',len(wordlist))

# # To create words cloud
# import wordcloud as wc
# font = '/Users/mingyuexu/PycharmProjects/demo/simhei.ttf'
# from imageio import imread
# backimg = imread('/Users/mingyuexu/PycharmProjects/demo/射雕背景1.png')
# wordscloud = wc.WordCloud(font_path=font,stopwords=set(stoplist),background_color=None,mode='RGBA',mask=backimg).generate(chapter5)
# wordscloud.to_file('/Users/mingyuexu/PycharmProjects/demo/wordcloud.png')
# fig = plt.figure()
# plt.imshow(wordscloud)
# plt.axis('off')
# plt.title('WordCloud')
# plt.show()

print('Now we introduce an amazing tool:')
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(stop_words=stoplist,min_df=5)
wordlist_up = [' '.join(wordlist_ind) for wordlist_ind in wordlist[0:10]]
print(wordlist_up[4])
result = CV.fit_transform(wordlist_up)
print(CV.vocabulary_)
print(result.todense())

print('distributed representation:')
