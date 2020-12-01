import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# 图像风格转换
# 超分辨率重构
# 图像修复

# Generative Adversarial Networks
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,BatchNormalization,Reshape,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from matplotlib.pyplot import subplot

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dims = 100

        optimizer = Adam(learning_rate=0.0002)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dims,))
        img = self.generator(z)

        # 在combined中，只训练生成器
        self.discriminator.trainable = False
        validation = self.discriminator(img)
        self.combined = Model(z,validation)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)


    def build_discriminator(self):
        model = Sequential(
            [
                Flatten(input_shape=self.img_shape),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dense(256),
                LeakyReLU(alpha=0.2),
                Dense(1),
                Activation('sigmoid')
            ]
        )
        img = Input(shape=self.img_shape)
        validation = model(img)
        return Model(img,validation)

    def build_generator(self):
        model = Sequential(
            [
                Dense(256,input_dim=self.latent_dims),
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(512),
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(1024),
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(np.prod(self.img_shape)),
                Activation('tanh'),
                Reshape(self.img_shape),
            ]
        )
        noise = Input(shape=(self.latent_dims,))
        img = model(noise)
        return Model(noise,img)

    def train(self,n_iter,batch_size=128,sample_interval=500):
        S = np.load('mnist.npz')
        X_train = S['x_train']
        print(X_train.shape)
        X_train = X_train/127.5 - 1
        X_train = np.expand_dims(X_train,axis=3)
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        for item in range(n_iter):
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0,1,(batch_size,self.latent_dims))
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,fake)
            d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
            noise = np.random.normal(0,1,(batch_size,self.latent_dims))
            g_loss = self.combined.train_on_batch(noise,valid)
            print('D loss:',d_loss)
            print('G loss:',g_loss)
            if item % sample_interval == 0:
                self.sample_images(item)

    def sample_images(self,n_iter):
        row,column = 5,5
        noise = np.random.normal(0,1,(row * column,self.latent_dims))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = gen_imgs * 0.5 + 0.5
        fig = plt.figure()
        for item in range(1,row * column+1,1):
            subplot(row,column,item)
            plt.imshow(gen_imgs[item-1,:,:,0],cmap='gray')
            plt.axis('off')
        fig.savefig(f'/Users/mingyuexu/PycharmProjects/demo/learning/Images/image{n_iter}')
        plt.close()

gan = GAN()
gan.train(n_iter=50000,batch_size=32,sample_interval=500)
