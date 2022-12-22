## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import gzip
import urllib.request

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
from tensorflow.keras.models import load_model
sys.path.append('../')


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNIST:
    def __init__(self):
        if not os.path.exists("../data"):
            os.mkdir("../data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]
        self.min_v = -0.5
        self.max_v = 0.5


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

    def evaluate(self, x, y):
        predicted = self.model(x).eval()
        acc = np.count_nonzero(predicted.argmax(1) == y.argmax(1)) / y.shape[0]
        return acc


class MNISTAEModel:
    def __init__(self, restore, session=None):
        params = [32, 64, 64, 49, 7, 64, 64, 32, 1]
        inp = Input((28, 28, 1))
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
        encoder = Model(inp, l1)
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
