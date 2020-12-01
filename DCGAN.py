import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

# Deep Convolution Generative Adversarial Networks
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Reshape,Dropout,Input,BatchNormalization,Activation,ZeroPadding2D,Conv2D,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from matplotlib.pyplot import subplot

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dims = 100

        optimizer = Adam(0.0002)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dims,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validation = self.discriminator(img)
        self.combined = Model(z,validation)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)

# padding='same'表示先填充再池化，而padding='valid'则是池化后舍弃一部分尾部，一般为了保护信息完整性选择'same'
# ZeroPadding: 如果为2个整数的2个元组:解释为((top_pad,bottom_pad),(left_pad,right_pad))
    def build_discriminator(self):
        model = Sequential(
            [
                Conv2D(32,kernel_size=(3,3),strides=(2,2),input_shape=self.img_shape,padding='same'),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same'),
                ZeroPadding2D(padding=((0,1),(0,1))),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(128, kernel_size=(3,3),strides=(2,2),padding='same'),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(256, kernel_size=(3,3),strides=(1,1),padding='same'),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Flatten(),
                Dense(1),
                Activation('sigmoid')
            ]
        )
        img = Input(shape=self.img_shape)
        validation = model(img)
        return Model(img,validation)

# UpSampling2D()是反池化层，可以把生成器理解成反卷积
    def build_generator(self):
        model = Sequential(
            [
                Dense(7*7*128,input_dim=self.latent_dims),
                Activation('relu'),
                Reshape((7,7,128)),
                UpSampling2D(),
                Conv2D(128,kernel_size=(3,3),padding='same'),
                BatchNormalization(momentum=0.8),
                Activation('relu'),
                UpSampling2D(),
                Conv2D(64,kernel_size=(3,3),padding='same'),
                BatchNormalization(momentum=0.8),
                Activation('relu'),
                Conv2D(self.channels,kernel_size=(3,3),padding='same'),
                Activation('tanh')
            ]
        )
        noise = Input(shape=(self.latent_dims,))
        img = model(noise)
        return Model(noise,img)

    def train(self,n_iter,batch_size,sample_interval=500):
        S = np.load('mnist.npz')
        X_train = S['x_train']
        X_train = X_train/127.5 - 1
        X_train = np.expand_dims(X_train,axis=3)
        true = np.ones((batch_size,1))
        false = np.zeros((batch_size,1))
        for item in range(1,n_iter+1,1):
            index = np.random.randint(0,X_train.shape[0],batch_size)
            true_img = X_train[index]
            noises = np.random.normal(0,1,(batch_size,self.latent_dims))
            false_img = self.generator.predict(noises)
            d_true_loss = self.discriminator.train_on_batch(true_img,true)
            d_false_loss = self.discriminator.train_on_batch(false_img,false)
            d_loss = np.add(d_true_loss,d_false_loss) * 0.5
            noises = np.random.normal(0, 1, (batch_size, self.latent_dims))
            g_loss = self.combined.train_on_batch(noises,true)
            print(f'd_loss={d_loss},while g_loss={g_loss}')
            if item % sample_interval == 0:
                self.sample_images_save(item)

    def sample_images_save(self,iter):
        fig = plt.figure()
        noises = np.random.normal(0, 1, (25, self.latent_dims))
        gen_imgs = self.generator.predict(noises)
        gen_imgs = gen_imgs * 0.5 + 0.5
        for item in range(1,26,1):
            subplot(5,5,item)
            plt.imshow(gen_imgs[item-1,:,:,0],cmap='gray')
            plt.axis('off')
        fig.savefig(f'/Users/mingyuexu/PycharmProjects/demo/learning/Images_DCGAN/image{iter}')
        plt.close()

DCgan = DCGAN()
DCgan.train(n_iter=10000,batch_size=32,sample_interval=100)
# 可以看到DCGAN效果要远好于普通的GAN
