import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

print('Text Classification:')
path_train_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.train.txt'
path_test_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.test.txt'
path_val_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.val.txt'

path_seg_train_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.train.seg.txt'
path_seg_test_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.test.seg.txt'
path_seg_val_file = '/Users/mingyuexu/PycharmProjects/demo/cnews.val.seg.txt'

path_vocabulary = '/Users/mingyuexu/PycharmProjects/demo/cnews.vocabulary.txt'
path_category = '/Users/mingyuexu/PycharmProjects/demo/cnews.category.txt'

import jieba as jb
def generate_seg_word(filename,output_filename):
    f = open(filename,mode='r')
    o_f = open(output_filename,mode='w')
    for line in f.readlines():
        label,content = line.strip('\r\n').split('\t')
        word_iter = jb.cut(content)
        word_content = ''
        for word in word_iter:
            word = word.strip(' ')
            if word != '':
                word_content += word + ' '
        output = '%s\t%s\n' % (label,word_content.strip(' '))
        o_f.writelines(output)
    o_f.close()
    f.close()

# generate_seg_word(path_train_file,path_seg_train_file)
# generate_seg_word(path_test_file,path_seg_test_file)
# generate_seg_word(path_val_file,path_seg_val_file)
# The above codes only need to run once.

f1 = open(path_seg_train_file,mode='r')
word_frequency = {}
for line in f1.readlines():
    label,content = line.strip('\r\n').split('\t')
    for item in content.split():
        if item not in word_frequency.keys():
            word_frequency[item] = 1
        else:
            word_frequency[item] += 1
word_frequency_list = sorted(word_frequency.items(),key=lambda x:x[1],reverse=True)
word_frequency_list = [item[0]+'@~'+str(item[1]) for item in word_frequency_list]
f2 = open(path_vocabulary,mode='w')
f2.writelines('<UNK>'+'@~'+str(10000000)+'\n')
for item in word_frequency_list:
    f2.writelines(item+'\n')
f2.close()
f1.close()

f3 = open(path_seg_train_file,mode='r')
category_dict = {}
for line in f3.readlines():
    label,content = line.strip('\r\n').split('\t')
    if label not in category_dict.keys():
        category_dict[label] = 1
    else:
        category_dict[label] += 1
category_dict_list = [item[0]+'@~'+str(item[1]) for item in category_dict.items()]
f4 = open(path_category,mode='w')
for item in category_dict_list:
    f4.writelines(item+'\n')
f4.close()
f3.close()

embedding_size = 32
time_steps = 600
word_threshold = 10
word_to_id = {}
category_to_id = {}

def vocabulary_to_dict(filename):
    unk = -1
    f = open(filename,mode='r')
    for line in f.readlines():
        list1 = line.split(sep='@~')
        word,frequency = list1[0],list1[1].strip('\n')
        frequency = int(frequency)
        if frequency < word_threshold:
            continue
        idx = len(word_to_id)
        if word == '<UNK>':
            unk = idx
        word_to_id[word] = idx
    f.close()
    return unk
unk = vocabulary_to_dict(path_vocabulary)
print(len(word_to_id))

def category_to_dict(filename):
    f = open(filename,mode='r')
    for line in f.readlines():
        list2 = line.split(sep='@~')
        category,num = list2[0],list2[1]
        category_to_id[category] = len(category_to_id)
    f.close()

category_to_dict(path_category)
print(category_to_id)

def sentence_to_idx(sentence):
    word_idxes = []
    for item in sentence:
        if item not in word_to_id.keys():
            word_idxes.append(unk)
        else:
            word_idxes.append(word_to_id[item])
    return word_idxes

from keras.utils import to_categorical
def data_prepocessing(filename):
    X = []
    Y = []
    f = open(filename,mode='r')
    for line in f.readlines():
        label, content = line.strip('\r\n').split('\t')
        id_label = category_to_id[label]
        id_content = sentence_to_idx(content)
        if len(id_content) >= time_steps:
            id_content = id_content[0:time_steps]
        else:
            id_content.extend([1 for i in range(time_steps-len(id_content))])
        X.append(id_content)
        Y.append(id_label)
    return np.array(X),to_categorical(np.array(Y),num_classes=10)

train_X,train_Y = data_prepocessing(path_seg_train_file)
test_X,test_Y = data_prepocessing(path_seg_test_file)
val_X,val_Y = data_prepocessing(path_seg_val_file)
print(train_X.shape,train_Y.shape)
print(test_X.shape,test_Y.shape)
print(val_X.shape,val_Y.shape)

print('------------------------------------')
print('Apply the LSTM:')
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,LSTMCell,RNN,Embedding
from keras.initializers import glorot_uniform,random_uniform

cells = [LSTMCell(64,kernel_initializer=glorot_uniform(),dropout=0.2),LSTMCell(64,kernel_initializer=glorot_uniform(),dropout=0.2)]
model = Sequential(
    [
        Embedding(len(word_to_id),embedding_size,embeddings_initializer=random_uniform(),input_length=600),
        RNN(cells),
        Dense(64),
        Activation('relu'),
        Dropout(0.2),
        Dense(10),
        Activation('softmax')
    ]
)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_X,train_Y,batch_size=100,epochs=100)
result_train = model.predict_classes(train_X)
from sklearn import metrics
print('accuracy for train data:',metrics.accuracy_score(y_true=train_Y,y_pred=result_train))
result_test = model.predict_classes(test_X)
print('accuracy for test data:',metrics.accuracy_score(y_true=test_Y,y_pred=result_test))
result_val = model.predict_classes(val_X)
print('accuracy for val data:',metrics.accuracy_score(y_true=val_Y,y_pred=result_val))

fig1 = plt.figure()
plt.plot(history.history['loss'],color='blue')
plt.title('loss')
plt.show()
