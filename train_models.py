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


def train_cnn_c100(data, file_name, input_shape, params, num_epochs=50, batch_size=128, train_temp=1, init=None, lr=0.01):
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
        Activation('softmax')
    ]
    model = Sequential()
    for layer in layers:
        model.add(layer)

    if init is not None:
        model.load_weights(init)

    loss_function = sparse_categorical_crossentropy
    optimizer = Adam()
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    shutil.rmtree('models/checkpoints')
    os.mkdir('models/checkpoints')
    filepath = 'models/checkpoints/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit(data.train_data, data.train_labels.argmax(1),
                        batch_size=batch_size,
                        validation_data=(data.validation_data, data.validation_labels.argmax(1)),
                        epochs=num_epochs,
                        callbacks=[checkpoint],
                        shuffle=True)
    print(history.history['val_loss'])
    val_loss = np.array(history.history['val_loss'])
    best_val_loss = 1000
    for file_name in os.listdir('models/checkpoints'):
        model.load_weights('models/checkpoints/' + file_name)
        loss = model.evaluate(data.test_data, data.test_labels.argmax(1), batch_size=128)[0]
        if loss < best_val_loss:
            best_val_loss = loss
            cp = 'models/checkpoints/' + file_name
    # model.load_weights(cp)
    if file_name != None:
        model.save_weights(file_name)

    results = model.evaluate(data.test_data, data.test_labels.argmax(1), batch_size=128)
    print("test loss, test acc:", results)
    y_pred = model.predict(data.test_data, batch_size=128)
    matrix = metrics.confusion_matrix(data.test_labels.argmax(1), y_pred.argmax(axis=1))
    print(matrix)

    return model



def train_mlp(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    input_shape = 29 * 29
    layers = [
        Dense(params[0], input_shape=(input_shape,)),
        Activation('relu'),
        Dropout(0.5),
        Dense(params[1]),
        Activation('relu'),
        Dense(params[2]),
        # Activation('softmax')
    ]
    model = Sequential()
    for layer in layers:
        model.add(layer)

    print(model.summary())

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted / train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    train_data = data.train_data.reshape((-1, 29 * 29))
    test_data = data.test_data.reshape((-1, 29 * 29))
    validation_data = data.validation_data.reshape((-1, 29 * 29))

    model.fit(train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save_weights(file_name)

    results = model.evaluate(test_data, data.test_labels, batch_size=128)
    print("test loss, test acc:", results)
    y_pred = model.predict(test_data, batch_size=128)
    matrix = metrics.confusion_matrix(data.test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    return model


def train_cnn_model_cicids_0(data, file_name) -> K_Model:
    # Creating layers
    inputs = Input(shape=(9, 9, 1))
    x = Conv2D(120, 2, activation='relu', padding="same")(inputs)
    x = Conv2D(60, 3, activation='relu', padding="same")(x)
    x = Conv2D(30, 4, activation='relu', padding="same")(x)
    x = Flatten()(x)
    outputs = Dense(13)(x)
    cnn_model = K_Model(inputs=inputs, outputs=outputs, name='cnn')

    def fn(correct, predicted):
        return tf.keras.losses.sparse_categorical_crossentropy(correct,
                                                               predicted, from_logits=True)

    cnn_model.compile(loss=fn,
                      metrics=['sparse_categorical_accuracy'],
                      optimizer='adam')

    # Checkpoint
    cp_path = os.path.join("models/checkpoints_cicids",
                           "5_2_cnn_weights-improvement-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(cp_path, monitor='val_sparse_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    logging.info("*** TRAINING START ***")
    history = cnn_model.fit(data.train_data, data.train_labels_indices,
                            validation_split=0.1, epochs=10, batch_size=1024, verbose=True)
    logging.info("*** TRAINING FINISH ***")
    if file_name is not None:
        cnn_model.save_weights(file_name)
    evaluation(cnn_model, data.test_data, data.test_labels_indices)
    return cnn_model


def train_cnn_model_cicids(data, file_name, params, num_epochs=10) -> K_Model:

    input_shape = (9, 9, 1)
    layers = [
        Conv2D(params[0], (2, 2), input_shape=input_shape, padding="same"),
        Activation('relu'),
        Conv2D(params[1], (3, 3), padding="same"),
        Activation('relu'),
        Conv2D(params[2], (4, 4), padding="same"),
        Activation('relu'),
        Flatten(),
        Dense(params[3]),
        Activation('relu'),
        Dense(params[4]),
        # Activation('softmax')
    ]
    cnn_model = Sequential()
    for layer in layers:
        cnn_model.add(layer)

    def fn(correct, predicted):
        return tf.keras.losses.sparse_categorical_crossentropy(correct,
                                                               predicted, from_logits=True)
    cnn_model.compile(loss=fn,
                      metrics=['sparse_categorical_accuracy'],
                      optimizer='adam')
    # Checkpoint
    cp_path = os.path.join("models/checkpoints_cicids",
                           "5_2_cnn_weights-improvement-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(cp_path, monitor='val_sparse_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    logging.info("*** TRAINING START ***")
    history = cnn_model.fit(data.train_data, data.train_labels_indices,
                            validation_split=0.1, epochs=num_epochs, batch_size=1024, verbose=True)
    logging.info("*** TRAINING FINISH ***")
    if file_name is not None:
        cnn_model.save_weights(file_name)
    evaluation(cnn_model, data.test_data, data.test_labels_indices)
    return cnn_model


def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.
    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name + "_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name + "_init", params, 1, batch_size)

    # now train the teacher at the given temperature
    teacher = train(data, file_name + "_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name + "_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted / train_temp))
        print(y)
        data.train_labels = y

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name + "_init")

    # and finally we predict at temperature 1
    predicted = student.predict(data.train_data)

    print(predicted)


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


def train_AE_CICIDS_1(data, file_name, input_shape, num_epochs=50, batch_size=128, train_temp=1, init=None):

    params = [120, 60, 30, 81, 9, 30, 60, 120, 1]
    inp = Input(input_shape)
    e = Conv2D(params[0], (2, 2), activation='relu', padding='same', name='conv2d')(inp)
    e = Conv2D(params[1], (3, 3), activation='relu', padding='same', name='conv2d_1')(e)
    e = Conv2D(params[2], (4, 4), activation='relu', padding='same', name='conv2d_2')(e)

    l = Flatten()(e)
    l1 = Dense(params[3])(l)
    l2 = Activation('softmax')(l1)
    d = Reshape((params[4], params[4], 1))(l2)

    d = Conv2DTranspose(params[5], (4, 4), activation='relu', padding='same', name='conv2dt')(d)
    d = Conv2DTranspose(params[6], (3, 3), activation='relu', padding='same', name='conv2dt_1')(d)
    d = Conv2DTranspose(params[7], (2, 2), activation='relu', padding='same', name='conv2dt_2')(d)

    decoded = Conv2D(params[8], (3, 3), activation='sigmoid', padding='same', name='conv2d_3')(d)
    ae_model = K_Model(inp, decoded)
    ae_model.summary()

    ae_model.compile(optimizer="adam", loss="mse")
    ae_model.fit(data.train_data, data.train_data,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 validation_data=(data.validation_data, data.validation_data),
                 shuffle=True,
                 verbose=1)

    if file_name is not None:
        ae_model.save_weights(file_name)

    return ae_model


def train_AE_CAN(data, file_name, input_shape, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    train_data = data.train_data
    inp = Input(input_shape)
    e = Conv2D(params[0], (3, 3), activation='relu', padding='same')(inp)
    e = Conv2D(params[1], (3, 3), activation='relu', padding='same')(e)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(params[2], (3, 3), activation='relu', padding='same')(e)
    e = Conv2D(params[3], (3, 3), activation='relu', padding='same')(e)
    e = MaxPooling2D((2, 2))(e)
    l = Flatten()(e)
    l1 = Dense(params[4])(l)
    l2 = Activation('softmax')(l1)
    d = Reshape((params[5], params[5], 1))(l2)
    d = Conv2DTranspose(params[6], (3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(params[7], (3, 3), activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(params[8], (3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(params[9], (3, 3), activation='relu', padding='same')(d)
    decoded = Conv2D(params[10], (3, 3), activation='sigmoid', padding='same')(d)
    ae_model = K_Model(inp, decoded)
    ae_model.summary()

    ae_model.compile(optimizer="adam", loss="mse")
    ae_model.fit(data.train_data, data.train_data,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 validation_data=(data.validation_data, data.validation_data),
                 shuffle=True,
                 verbose=1)

    if file_name is not None:
        ae_model.save_weights(file_name)

    return ae_model


def load_CIFAR():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_val = x_test[:7000]
    x_test = x_test[7000:]


def trainAE1(data, file_name, num_epochs=50, batch_size=32, train_temp=1, init=None):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    l = Flatten()(encoded)

    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    model = K_Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    chkpt = 'models/AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(data.train_data, data.train_data,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=(data.validation_data, data.validation_data),
                        callbacks=[es_cb, cp_cb],
                        shuffle=True)

    if file_name is not None:
        model.save_weights(file_name)

    return model


def fea_correlation(data):
    data_0 = data.train_data.reshape((-1, 81))
    fea_num = 78
    data_new = data_0[:, :fea_num].copy()
    corr_matrix = np.ones((fea_num, fea_num))
    for i in range(fea_num):
        print('feature: {}'.format(i))
        for j in range(i, fea_num):
            if j==i:
                continue
            else:
                corr_matrix[i, j], _ = pearsonr(data_new[:, i], data_new[:, j])
            corr_matrix[j, i] = corr_matrix[i, j]
        print(corr_matrix[i, :])
    np.save('./figures/correlations.npy', corr_matrix)
    return corr_matrix


def plot_corr(data):
    data_0 = data.train_data.reshape((-1, 81))
    fea_num = 78
    rows = list(range(30)) + list(range(34, 43)) + list(range(51, 56)) + list(range(62, 70))
    rows = list(range(30))
    data_new = data_0[:, rows].copy()
    df = pd.DataFrame(data_new)
    # corrMatrix = df.corr()
    # print(corrMatrix)
    # sn.heatmap(corrMatrix, annot=True)
    # plt.show()

    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


if not os.path.isdir('models'):
    os.makedirs('models')

# attack_class = [('DoS', 0.0)]
# attack_class = [('Probe', 2.0)]
# attack_class = [('R2L', 3.0)]
# attack_class = [('U2R', 4.0)]

# train(NSL_KDD(attack_class), "models/nsl_kdd_Dos.h5", [50], num_epochs=50)
# train(NSL_KDD(attack_class), "models/nsl_kdd_Probe.h5", [50], num_epochs=20)
# train(NSL_KDD(attack_class), "models/nsl_kdd_R2L.h5", [50], num_epochs=20)
# train(NSL_KDD(attack_class), "models/nsl_kdd_U2R.h5", [50], num_epochs=20)

# def tr(v):
#     # tensorflow weights to pytorch weights
#     if v.ndim == 4:
#         return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
#     elif v.ndim == 2:
#         return np.ascontiguousarray(v.transpose())
#     return v
#
# def read_ckpt(ckpt):
#     # https://github.com/tensorflow/tensorflow/issues/1823
#     reader = tf.train.load_checkpoint(ckpt)
#     weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
#     pyweights = {k: tr(v) for (k, v) in weights.items()}
#     return pyweights
#
# weights = read_ckpt('models/nsl_kdd_Dos.h5')
# dd.io.save('models/nsl_kdd_Dos_weights.h5', weights)

# model = NSLModel('models/nsl_kdd_Dos.h5', 121)
# weights = model.model.get_weights()
#
# for i, w in enumerate(weights):
#     np.save('models/nsl_kdd_Dos_weights_l{}.npy'.format(i), w)
#
# weight_loaded = np.load('models/nsl_kdd_Dos_weights_l1.npy')

# train_cnn(CIFAR(), "models/cifar.h5", [64, 64, 128, 128, 256, 256, 10], num_epochs=50)
# train_cnn(MNIST(), "models/mnist.h5", [32, 32, 64, 64, 200, 200, 10], num_epochs=30)
# train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200, 10],
#                    num_epochs=50, train_temp=100)
# train_distillation(CIFAR(), "models/cifar-distilled-100", [64, 64, 128, 128, 256, 256, 10],
#                    num_epochs=50, train_temp=100)

#
# trainAE(MNIST(), "./models/mnist_ae.h5", (28,28,1), [32, 64, 64, 49, 7, 64, 64, 32, 1], num_epochs=10)
# trainAE(CIFAR(), "./models/cifar_ae.h5", (32, 32, 3), [64, 64, 128, 64, 8, 128, 64, 64, 3], num_epochs=3)
# trainAE1(CIFAR(), "./models/cifar_ae.h5", num_epochs=10)

np.random.seed(1)
tf.random.set_seed(0)
# train_cnn_model_cicids(CICIDS(), "./models/cicids.h5", params = [120, 60, 30, 50, 13], num_epochs=10)
# train_cnn_model_cicids(CICIDS(attack_cat=10), "./models/cicids_binary.h5", params=[120, 60, 30, 50, 2], num_epochs=2)
# train_AE_CICIDS_1(CICIDS(), './models/CICIDSAE.h5', input_shape=(9, 9, 1), num_epochs=10)
# correlations = fea_correlation(CICIDS())
# print(correlations)
# plot_corr(CICIDS())

# train_cnn(CIFAR100(3), "models/c100.h5", (32, 32, 3), [64, 64, 128, 128, 256, 256, 3], batch_size=32, num_epochs=20, lr=0.005)
# train_cnn_c100(CIFAR100(3), "models/c100.h5", (32, 32, 3), [64, 64, 128, 128, 256, 256, 3], batch_size=32, num_epochs=20, lr=0.005)
trainAE(CIFAR100(), "./models/c100_ae.h5", (32,32,3), [64, 64, 128, 64, 8, 128, 64, 64, 3], num_epochs=10)






# # CAN bus
# seed = 3
# tf.random.set_seed(seed)
# np.random.seed(seed)
# # can_pre_process_2_classes()
# # possible attack type: DoS Fuzzy gear RPM
# attack_type = 'RPM'
# can_data = CAN(attack_type)
# train_data = can_data.train_data
# train_labels = can_data.train_labels
#
# print('shape of train data', train_data.shape)
# print('shape of train labels', train_labels.shape)
# train_cnn(can_data, 'models/can_{}_1.h5'.format(attack_type), (28, 28, 1),
#           [64, 64, 128, 128, 256, 256, 2], num_epochs=15)
# # train_AE_CAN(can_data, "./models/can_{}_ae_1.h5".format(attack_type), (28, 28, 1),
# #              [64, 64, 128, 128, 49, 7, 128, 128, 64, 64, 1], num_epochs=10)
#
# # train_mlp(can_data, "models/Fuzzy.h5", [1, 1, 2], num_epochs=2)


#
# def train_cnn():
#     model = Sequential()
#
#     print(data.train_data.shape)
#
#     model.add(Conv2D(params[0], (3, 3),
#                      input_shape=data.train_data.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(Conv2D(params[1], (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(params[2], (3, 3)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(params[3], (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(params[4]))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(params[5]))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
