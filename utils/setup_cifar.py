## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
from tensorflow.keras.models import load_model

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []
        
        if not os.path.exists("../data/cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()

        for i in range(5):
            r,s = load_batch("cdata/ifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch("data/cifar-10-batches-bin/test_batch.bin")
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

class CIFARModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)


class CIFARAEModel:
    def __init__(self, restore, session=None):
        params = [64, 64, 128, 64, 8, 128, 64, 64, 3]
        inp = Input((32, 32, 3))
        e = Conv2D(params[0], (3, 3), activation='relu')(inp)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(params[1], (3, 3), activation='relu')(e)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(params[2], (3, 3), activation='relu')(e)
        l = Flatten()(e)
        l1 = Dense(params[3])(l)
        l2 = Activation('softmax')(l1)
        d = Reshape((params[4], params[4], 1))(l2)
        d = Conv2DTranspose(params[5], (3, 3), strides=2, activation='relu',
                            padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(params[6], (3, 3), strides=2, activation='relu',
                            padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(params[7], (3, 3), activation='relu', padding='same')(d)
        decoded = Conv2D(params[8], (3, 3), activation='sigmoid', padding='same')(d)
        model = Model(inp, decoded)
        encoder = Model(inp, l)
        # decoder = Model(l, decoded)
        # model.load_weights(restore).expect_partial()
        # encoder.load_weights(restore).expect_partial()
        model.load_weights(restore)
        encoder.load_weights(restore, by_name=True)
        # decoder.load_weights(restore, by_name=True)
        # weights_list = model.get_weights()
        # for i, weights in enumerate(weights_list[0:5]):
        #     encoder.layers[i].set_weights(weights)

        self.model = model
        self.encoder = encoder
        # self.decoder = decoder

    def predict(self, data):
        return self.model(data)

    def get_latent(self, data):
        return self.encoder(data)


# class CIFARAEModel:
#     def __init__(self, restore, session=None):
#         input_img = Input(shape=(32, 32, 3))
#         x = Conv2D(64, (3, 3), padding='same')(input_img)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), padding='same')(x)
#         x = Conv2D(32, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D((2, 2), padding='same')(x)
#         x = Conv2D(16, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         encoded = MaxPooling2D((2, 2), padding='same')(x)
#         l = Flatten()(encoded)
# 
#         x = Conv2D(16, (3, 3), padding='same')(encoded)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = UpSampling2D((2, 2))(x)
#         x = Conv2D(32, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = UpSampling2D((2, 2))(x)
#         x = Conv2D(64, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = UpSampling2D((2, 2))(x)
#         x = Conv2D(3, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         decoded = Activation('sigmoid')(x)
# 
#         model = Model(input_img, decoded)
#         encoder = Model(input_img, l)
#         model.load_weights(restore)
#         encoder.load_weights(restore, by_name=True)
#         self.model = model
#         self.encoder = encoder
# 
#     def predict(self, data):
#         return self.model(data)
# 
#     def get_latent(self, data):
#         return self.encoder(data)