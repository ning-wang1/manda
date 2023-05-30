# train_models.py -- train the neural network models for attacking
# Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

import json
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Model as K_Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn import metrics
import numpy as np
import shutil
import logging
import pandas as pd
from utils.setup_cicids import evaluation, CICIDS
from scipy.stats import pearsonr
import seaborn as sn
import matplotlib.pyplot as plt
import deepdish as dd
# from utils.util_artifact import get_model

from utils.setup_NSL import NSL_KDD, NSLModel
from utils.setup_mnist import MNIST
from utils.setup_can import can_pre_process_2_classes, CAN
from utils.setup_coco import COCO, CIFAR100

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Log setting
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

# Change display.max_rows to show all features.
pd.set_option("display.max_rows", 85)


def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None, lr=0.01):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)
    model.add(Dense(params[0], input_dim=data.train_data.shape[1], activation='relu'))
    model.add(Dense(2))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted / train_temp)

    sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    y_pred = model.predict(data.test_data, batch_size=128)
    matrix = metrics.confusion_matrix(data.test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    return model


def train_cnn(data, file_name, input_shape, params, num_epochs=50, batch_size=128, train_temp=1, init=None, lr=0.01):
    """
    Standard neural network training procedure.
    """
    layers = [
        Conv2D(params[0], (3, 3), input_shape=input_shape),
        Activation('relu'),
        Conv2D(params[1], (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(params[2], (3, 3)),
        Activation('relu'),
        Conv2D(params[3], (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(params[4]),
        Activation('relu'),
        Dropout(0.5),
        Dense(params[5]),
        Activation('relu'),
        Dense(params[6]),
        # Activation('softmax')
    ]
    model = Sequential()
    for layer in layers:
        model.add(layer)

    if init is not None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted / train_temp)

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # checkpoint_path = "models/checkpoints/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # shutil.rmtree('models/checkpoints')
    # os.mkdir('models/checkpoints')
    #
    # period = 1
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  verbose=1,
    #                                                  save_weights_only=True,
    #                                                  period=period)
    # model.save_weights(checkpoint_path.format(epoch=0))

    shutil.rmtree('models/checkpoints')
    os.mkdir('models/checkpoints')
    filepath = 'models/checkpoints/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit(data.train_data, data.train_labels,
                        batch_size=batch_size,
                        validation_data=(data.validation_data, data.validation_labels),
                        epochs=num_epochs,
                        callbacks=[checkpoint],
                        shuffle=True)
    print(history.history['val_loss'])
    val_loss = np.array(history.history['val_loss'])
    best_val_loss = 1000
    for file_name in os.listdir('models/checkpoints'):
        model.load_weights('models/checkpoints/' + file_name)
        loss = model.evaluate(data.test_data, data.test_labels, batch_size=128)[0]
        if loss < best_val_loss:
            best_val_loss = loss
            cp = 'models/checkpoints/' + file_name
    model.load_weights(cp)
    if file_name != None:
        model.save_weights(file_name)

    results = model.evaluate(data.test_data, data.test_labels, batch_size=128)
    # results = model.evaluate(data.train_data, data.train_labels, batch_size=128)
    print("test loss, test acc:", results)
    y_pred = model.predict(data.test_data, batch_size=128)
    matrix = metrics.confusion_matrix(data.test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    return model


def trainAE(data, file_name, input_shape, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    train_data = data.train_data
    inp = Input(input_shape)
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
    ae_model = K_Model(inp, decoded)
    ae_model.summary()

    ae_model.compile(optimizer="adam", loss="mse")
    ae_model.fit(data.train_data, data.train_data,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 validation_data=(data.validation_data, data.validation_data),
                 verbose=1
                 )

    if file_name is not None:
        ae_model.save_weights(file_name)

    return ae_model


if __name__=='__main__':
    if not os.path.isdir('models'):
        os.makedirs('models')

    attack_class = [('DoS', 0.0)]

    # train(NSL_KDD(attack_class), "models/nsl_kdd_Dos.h5", [50], num_epochs=50)

    train_cnn(MNIST(), "models/mnist.h5", (28,28,1), [32, 32, 64, 64, 200, 200, 10], num_epochs=30)

    np.random.seed(1)
    tf.random.set_seed(0)
    # train_cnn_model_cicids(CICIDS(), "./models/cicids.h5", params = [120, 60, 30, 50, 13], num_epochs=10)
    # train_cnn_model_cicids(CICIDS(attack_cat=10), "./models/cicids_binary.h5", params=[120, 60, 30, 50, 2], num_epochs=2)


