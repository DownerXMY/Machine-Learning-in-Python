import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

print('Seq to seq:')

df = pd.read_csv('/Users/mingyuexu/PycharmProjects/demo/fake_or_real_news.csv')
print(df['title'][0])
X = df['text']
Y = df['title']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

Max_Input_length = 500
Max_Target_length = 50
Max_Input_Vocabulary_Size = 5000
Max_Target_Vocabulary_Size = 2000
def fit_text(X,Y,input_length=None,target_length=None):
    if input_length == None:
        input_length = Max_Input_length
    if target_length == None:
        target_length = Max_Target_length
    input_counter = Counter()
    target_counter = Counter()
    max_input_length = 0
    max_target_length = 0

    for item in X:
        text = [word.lower() for word in item.split(' ')]
        seq_length = len(text)
        if seq_length > input_length:
            text = text[0:input_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_input_length = max(max_input_length,seq_length)

    for item in Y:
        item_up = 'START ' + item.lower() + ' END'
        title = [word for word in item_up.split(' ')]
        seq_length = len(title)
        if seq_length > target_length:
            title = title[0:target_length]
            seq_length = len(title)
        for word in title:
            target_counter[word] += 1
        max_target_length = max(max_target_length,seq_length)

    text2idx = dict()
    for idx,word in enumerate(input_counter.most_common(Max_Input_Vocabulary_Size)):
        text2idx[word[0]] = idx + 2
    text2idx['PAD'] = 0
    text2idx['UNK'] = 1
    idx2text = dict([(idx,word) for word,idx in text2idx.items()])

    target2idx = dict()
    for idx,word in enumerate(target_counter.most_common(Max_Target_Vocabulary_Size)):
        target2idx[word[0]] = idx + 1
    target2idx['UNK'] = 0
    idx2target = dict([(idx,word) for word,idx in target2idx.items()])

    num_input_tokens = len(text2idx)
    num_target_tokens = len(target2idx)
    config = dict()
    config['text2idx'] = text2idx
    config['target2idx'] = target2idx
    config['idx2text'] = idx2text
    config['idx2target'] = idx2target
    config['input_tokens'] = num_input_tokens
    config['target_tokens'] = num_target_tokens
    config['max_input_length'] = max_input_length
    config['max_target_length'] = max_target_length
    return config

config = fit_text(X,Y)

from keras.preprocessing.sequence import pad_sequences
def data_transformation(data):
    data_tran = []
    if data is X_train or data is X_test:
        max_length = config['max_input_length']
        dic = config['text2idx']
        for item in data:
            item_up = item.lower()
            data_got = [word for word in item_up.split(' ')]
            data_got_cor = []
            for word in data_got:
                if word not in dic.keys():
                    data_got_cor.append('UNK')
                else:
                    data_got_cor.append(word)
            data_trans = [dic[word] for word in data_got_cor]
            if len(data_trans) >= max_length:
                data_trans = data_trans[0:max_length]
            data_tran.append(data_trans)
        data_tran = pad_sequences(data_tran,maxlen=max_length,padding='post')
    if data is Y_train or data is Y_test:
        max_length = config['max_target_length']
        dic = config['target2idx']
        for item in data:
            item_up = 'START ' + item.lower() + ' END'
            data_got = [word for word in item_up.split(' ')]
            data_got_cor = []
            for word in data_got:
                if word not in dic.keys():
                    data_got_cor.append('UNK')
                else:
                    data_got_cor.append(word)
            data_trans = [dic[word] for word in data_got_cor]
            if len(data_trans) >= max_length:
                data_trans = data_trans[0:max_length]
            data_tran.append(data_trans)
        data_tran = np.array(data_tran)
    return data_tran,data_tran.shape

X_train,X_train_shape = data_transformation(X_train)
X_test,X_test_shape = data_transformation(X_test)
Y_train,Y_train_shape = data_transformation(Y_train)
Y_test,Y_test_shape = data_transformation(Y_test)
print(X_train_shape,X_test_shape,Y_train_shape,Y_test_shape)


from keras.models import Model,load_model
from keras.layers import Dense,Embedding,Input,LSTM
class Seq2Seq(object):
    def __init__(self,config,HIDDEN_UNITS=100,batch_size=64):
        self.config = config
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.batch_size = batch_size
        self.version = 0

        if 'version' in config:
            self.version = config['version']
        encoder_inputs = Input(shape=(None,),name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.config['input_tokens'],output_dim=self.HIDDEN_UNITS,
                                      input_length=self.config['max_input_length'],name='encoder_embedding')
        encoder_lstm = LSTM(units=self.HIDDEN_UNITS,return_state=True,name='encoder_lstm')
        encoder_outputs,encoder_state_h,encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h,encoder_state_c]

        decoder_inputs = Input(shape=(None,self.config['target_tokens']),name='decoder_inputs')
        decoder_lstm = LSTM(units=self.HIDDEN_UNITS,return_state=True,return_sequences=True,name='decoder_lstm')
        decoder_outputs,decoder_state_h,decoder_state_c = decoder_lstm(decoder_inputs,initial_state=encoder_states)
        decoder_dense = Dense(units=self.config['target_tokens'],activation='softmax',name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model_combined = Model([encoder_inputs,decoder_inputs],decoder_outputs)
        model_combined.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        self.model = model_combined
        self.encoder_model = Model(encoder_inputs,encoder_states)
        decoder_state_inputs = [Input(shape=(self.HIDDEN_UNITS,)),Input(shape=(self.HIDDEN_UNITS,))]
        decoder_outputs,state_h,state_c = decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)
        decoder_states = [state_h,state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs]+decoder_state_inputs,[decoder_outputs]+decoder_states)

    def fit(self):
        train_gen = self.generate_batch(X_train,Y_train,batch_size=self.batch_size)
        test_gen = self.generate_batch(X_test,Y_test,batch_size=self.batch_size)

        history = self.model.fit_generator(generator=train_gen,steps_per_epoch=len(X_train)//self.batch_size,
                                           epochs=100,validation_data=test_gen,
                                           validation_steps=len(X_test)//self.batch_size)
        self.model.save('/Users/mingyuexu/PycharmProjects/demo/learning/model_seq2seq.h5')
        self.encoder_model.save('/Users/mingyuexu/PycharmProjects/demo/learning/model_encoder.h5')
        self.decoder_model.save('/Users/mingyuexu/PycharmProjects/demo/learning/model_decoder.h5')
        return history

    def mypredict(self):
        pred_model = load_model('/Users/mingyuexu/PycharmProjects/demo/learning/model_seq2seq.h5')
        enc_model = load_model('/Users/mingyuexu/PycharmProjects/demo/learning/model_encoder.h5')
        dec_model = load_model('/Users/mingyuexu/PycharmProjects/demo/learning/model_decoder.h5')
        test_data = []
        true_result = []
        for item in np.random.permutation(np.arange(len(X)))[0:20]:
            test_x = X[item]
            true_headline = Y[item]
            true_result.append(true_headline)
            test_cor = []
            for word in test_x.lower().split(' '):
                if word not in self.config['text2idx'].keys():
                    test_cor.append('UNK')
                else:
                    test_cor.append(word)
            test_label = [self.config['text2idx'][word] for word in test_cor]
            test_data.append(test_label)
        test_data_cor = pad_sequences(test_data,maxlen=self.config['max_input_length'],padding='post')
        print(test_data_cor.shape)
        state_values = enc_model.predict(test_data_cor)
        target_seq = np.zeros((1,1,self.config['target_tokens']))
        target_seq[0,0,self.config['target2idx']['START']] = 1
        terminated = False
        target_length = 0
        pred_target = ''
        while not terminated:
            output_tokens,state_h,state_c = dec_model.predict([target_seq]+state_values)
            sample_token_idx = np.argmax(output_tokens[0,-1,:])
            sample_word = self.config['idx2target'][sample_token_idx]
            target_length += 1
            if sample_word != 'START' and sample_word != 'END':
                pred_target += ' ' + sample_word
            if sample_word == 'END' or target_length >= self.config['max_target_length']:
                terminated = True
            target_seq = np.zeros((1,1,self.config['target_tokens']))
            target_seq[0,0,sample_token_idx] = 1
            state_values = [state_h,state_c]
        return pred_target.strip(),true_result

    def generate_batch(self,X_samples,Y_samples,batch_size):
        n_iter = len(X_samples) // batch_size
        while True:
            for index in range(0,n_iter,1):
                start = index * batch_size
                end = (index + 1) * batch_size
                encoder_batch_input = X_samples[start:end]
                decoder_batch_target = np.zeros(shape=(batch_size,self.config['max_target_length'],self.config['target_tokens']))
                decoder_batch_input = np.zeros(shape=(batch_size,self.config['max_target_length'],self.config['target_tokens']))
                for number,target_labels in enumerate(Y_samples[start:end]):
                    for title_number,target_label in enumerate(target_labels):
                        if target_label != 0:
                            decoder_batch_input[number,title_number,target_label] = 1
                            if title_number > 0:
                                decoder_batch_target[number,title_number-1,target_label] = 1
                yield [encoder_batch_input,decoder_batch_input], decoder_batch_target

seq2seq = Seq2Seq(config)
# history = seq2seq.fit()
prediction,actual = seq2seq.mypredict()
print('0:',actual)
print('1:',prediction)