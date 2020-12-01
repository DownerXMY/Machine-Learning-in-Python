import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# 长短期记忆网络(LSTM)
# 双向循环神经网络(BRNN)
# 深层循环神经网络(DRNN)

print('simple RNN:')
import pandas as pd
df = pd.read_csv('/Users/mingyuexu/PycharmProjects/zgpa_train.csv')
print(df)
df = df.loc[:,['date','close']]
price = df.loc[:,'close']
price_norm = price / price.max()

plt.plot(price,color='blue',label='original')

def extract_data(data,time_step):
    X = []
    Y = []
    n = 0
    while n < len(data)-time_step+1:
        X.extend(data[n:n+time_step])
        n += 1
    X = np.array([X])
    X = X.reshape(len(data)-time_step+1,time_step,1)
    Y = np.array(data[time_step-1:])
    return  [X,Y]

train_data = extract_data(price_norm,time_step=8)
print(train_data[0].shape,train_data[1].shape)

from keras.models import Sequential
from keras.layers import Dense,Activation,SimpleRNN

model = Sequential(
    [
        SimpleRNN(5,input_shape=(8,1)),
        Activation('relu'),
        Dense(1),
        Activation('linear'),
    ]
)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(train_data[0],train_data[1],epochs=200)
result = model.predict(train_data[0])
result_rec = result * price.max()
from sklearn.metrics import r2_score, mean_squared_error
print('r2_score=',r2_score(y_true=train_data[1],y_pred=result))
print('mean_squared_error=',mean_squared_error(y_true=train_data[1],y_pred=result))

plt.plot(result_rec,color='red',label='predict')
plt.legend()
plt.title('close-price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
print('-----------------------------------')

print('LSTM:')
f = open('article.txt',mode='r').read()
f = f.replace('\n','').replace('\r','')
print(len(f))

letters = list(set(f))
print(letters)
num_letters = len(letters)
print(len(letters))

int_to_char = {
    a:b for a,b in enumerate(letters)
}
char_to_int = {
    b:a for a,b in enumerate(letters)
}
# print(char_to_int)

from keras.utils import to_categorical
def data_preprocessing(data,time_step,letters,char_to_int):
    n = 0
    X = []
    while n < len(data)-time_step:
        data1 = data[n:n+time_step]
        data_Xi = [char_to_int[item] for item in data1]
        X.extend(data_Xi)
        n += 1
    X = np.array([X])
    X.reshape(len(data)-time_step,time_step)
    X_up = to_categorical(X, num_classes=len(letters))
    data2 = data[time_step:]
    Y = np.array([char_to_int[item] for item in data2])
    return [X_up,Y]

[X_train,Y_train] = data_preprocessing(data=f,time_step=20,letters=letters,char_to_int=char_to_int)
X_train = X_train.reshape(11139,20,64)
Y_train_up = to_categorical(Y_train,num_classes=64)

from keras.layers import LSTM
model1 = Sequential(
    [
        LSTM(20,input_shape=(20,64)),
        Activation('relu'),
        Dense(64),
        Activation('softmax')
    ]
)
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.fit(X_train,Y_train_up,batch_size=1000,epochs=50)
result2 = model1.predict_classes(X_train)
print(result2)
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_true=Y_train,y_pred=result2))
Y_pred_txt = [int_to_char[item] for item in result2]
str1 = ''
for item in Y_pred_txt:
    str1 = str1 + item
print(str1)


