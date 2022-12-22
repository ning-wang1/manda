import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import random

from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
from tensorflow.keras.models import load_model
import logging

import pandas as pd
import tensorflow.keras as keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# Log setting
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

# Change display.max_rows to show all features.
pd.set_option("display.max_rows", 85)

sys.path.append('../')


def preprocessing(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # Shuffle the dataset
    df = df.sample(frac=1)

    # Split features and labels
    x = df.iloc[:, df.columns != 'Label']
    y = df[['Label']].to_numpy()

    # Scale the features between 0 ~ 1
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return x, y, scaler


def reshape_dataset_cnn(x: np.ndarray) -> np.ndarray:
    # Add padding columns
    result = np.zeros((x.shape[0], 81))
    result[:, :-3] = x

    # Reshaping dataset
    result = np.reshape(result, (result.shape[0], 9, 9))
    result = result[..., tf.newaxis]
    return result


def plot_history(history: tf.keras.callbacks.History):
    # summarize history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model2 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model2 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def evaluation(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    score = model.evaluate(x_test, y_test, verbose=False)
    logging.info('Evaluation:\nLoss: {}\nAccuracy : {}\n'.format(score[0], score[1]))

    # F1 score
    y_pred = model.predict(x_test, batch_size=1024, verbose=False)
    y_pred = np.argmax(y_pred, axis=1)

    logging.info("\n{}".format(classification_report(y_test, y_pred)))


def one_hot_coding(labels, class_num):
    labels_ls = list(labels)
    labels_ls = [int(y) for y in labels_ls]
    labels_one_hot = np.eye(class_num)[labels_ls]
    return labels_one_hot


class CICIDS:
    def __init__(self, test_num=50000, attack_cat=None):
        """ CICIDS includes 13 catogories, 1 benign and 12 attacks.
        Pram: attack_cat: if it is not None, we select a subset of data that belonging to the attack_cat
        """
        scale = 0.4
        class_num = 13
        if attack_cat is not None:
            class_num = 2

        # load train data
        train_data = pd.read_csv("data/CICIDS2017/ProcessedDataset/train_MachineLearningCVE.csv",
                                 skipinitialspace=True)
        logging.info("Class distribution\n{}".format(train_data.Label.value_counts()))
        X_train, Y_train, scaler = preprocessing(train_data)
        del train_data

        # select a specific attack
        if attack_cat is not None:
            pos1 = np.where(Y_train == attack_cat)[0]
            Y_train[pos1] = np.ones((len(pos1), 1))  # attach label 1 to the attack
            pos2 = np.where(Y_train == 0)[0]  # 0 represents benign class
            pos = np.concatenate((pos1, pos2), axis=0)
            pos = np.sort(pos)
            X_train = X_train[pos]
            Y_train = Y_train[pos]

        # manage the input data in 2-D (image like) format and set the range from -scale to scale
        X_train = reshape_dataset_cnn(X_train)
        X_train = X_train * scale
        Y_train_one_hot = one_hot_coding(Y_train, class_num)

        # load test data
        test_data = pd.read_csv('data/CICIDS2017/ProcessedDataset/test_MachineLearningCVE.csv',
                                skipinitialspace=True)
        logging.info("Class distribution\n{}".format(test_data.Label.value_counts()))
        X_test, Y_test, _ = preprocessing(test_data)
        del test_data

        if attack_cat is not None:
            pos1 = np.where(Y_test == attack_cat)[0]
            Y_test[pos1] = np.ones((len(pos1), 1))
            pos2 = np.where(Y_test == 0)[0]  # 0 represents benign class
            pos = np.concatenate((pos1, pos2), axis=0)
            pos = np.sort(pos)
            X_test = X_test[pos]
            Y_test = Y_test[pos]

        # manage the input data in 2-D (image like) format and set the range from -scale to scale
        X_test = reshape_dataset_cnn(X_test)
        X_test = X_test * scale
        Y_test_one_hot = one_hot_coding(Y_test, class_num)

        if test_num is None:
            test_num = len(Y_test)
        self.test_data = X_test[:test_num, :, :, :]
        self.test_labels = Y_test_one_hot[:test_num]
        self.test_labels_indices = Y_test[:test_num]
        self.scaler = scaler

        # split validation data
        VALIDATION_SIZE = 5000
        self.validation_data = X_train[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = Y_train_one_hot[:VALIDATION_SIZE]
        self.validation_labels_indices = Y_train[:VALIDATION_SIZE]
        self.train_data = X_train[VALIDATION_SIZE:, :, :, :]
        self.train_labels = Y_train_one_hot[VALIDATION_SIZE:]
        self.train_labels_indices = Y_train[VALIDATION_SIZE:]
        self.min_v = 0
        self.max_v = scale

        train_data_flatten = self.train_data.reshape((-1, 81))
        # self.feature_mean = np.mean(train_data_flatten, axis=0)
        class_num = self.train_labels.shape[1]
        self.feature_mean = np.zeros((class_num, 81))
        for i in range(class_num):
            idx = self.sel_target_class(i)[0]
            y = train_data_flatten[idx, :].copy()
            self.feature_mean[i, :] = np.mean(y, axis=0)
        # print(self.feature_mean)

    def sel_target_class(self, target_class):
        labels = np.argmax(self.train_labels, axis=1)
        idx = np.where(labels == target_class)
        return idx


class CICIDSModel:
    def __init__(self, restore, session=None, binary=False):
        self.num_channels = 1
        self.image_size = 9
        self.num_labels = 13
        if binary:
            self.num_labels = 2

        params = [120, 60, 30, 50, self.num_labels]
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
            Dense(params[4])]

        model = Sequential()
        for layer in layers:
            model.add(layer)

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

    def evaluate(self, x, y):
        predicted = self.model(x).eval()
        acc = np.count_nonzero(predicted.argmax(1) == y.argmax(1)) / y.shape[0]
        return acc

    def predict_classes(self, x):
        predicted = self.model(x).eval()
        y = predicted.argmax(1)
        return y


# class CICIDSAEModel:
#     def __init__(self, session=None):
#         self.input_shape = (9, 9, 1)
#
#     def predict(self, data):
#         return self.model(data)
#
#     def get_latent(self, data):
#         new_data = data.reshape((-1, 81))
#         return new_data


# class CICIDSAEModel:
#     def __init__(self, restore, session=None):
#         self.num_channels = 1
#         self.image_size = 9
#         self.num_labels = 13
#
#         params = [120, 60, 30, 50, 13]
#         input_shape = (9, 9, 1)
#         layers = [
#             Conv2D(params[0], (2, 2), input_shape=input_shape, padding="same"),
#             Activation('relu'),
#             Conv2D(params[1], (3, 3), padding="same"),
#             Activation('relu'),
#             Conv2D(params[2], (4, 4), padding="same"),
#             Activation('relu'),
#             Flatten(),
#             Dense(params[3]),
#             Activation('relu'),
#             Dense(params[4])]
#
#         model = Sequential()
#         for layer in layers:
#             model.add(layer)
#         encoder = Sequential()
#         for layer in layers[:-1]:
#             encoder.add(layer)
#
#         model.load_weights(restore)
#         encoder.load_weights(restore, by_name=True, skip_mismatch=True)
#         self.model = model
#
#         self.model = model
#         self.encoder = encoder
#
#     def predict(self, data):
#         return self.model(data)
#
#     def get_latent(self, data):
#         return self.encoder(data)


class CICIDSAEModel:
    def __init__(self, restore, session=None):
        params = [120, 60, 30, 81, 9, 30, 60, 120, 1]
        inp = Input((9, 9, 1))
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


def sel_target_class(target_class, train_labels):
    train_labels = np.argmax(train_labels, axis=1)
    y = map(lambda x: x if x == target_class else 100, train_labels)
    y_new = np.array([item for item in y])
    idx = [i for i in range(1800000) if y_new[i] != 100]
    return idx


def subset_dataset(train_data, train_labels, num=15000):
    input_shape = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    sample_num = num  # cicids
    seldom_class = [1, 8, 9, 11, 12]
    remain_class = list(set(list(range(13))) - set(seldom_class))
    idx_seldom_class = []
    sample_num_per_class = int(sample_num / len(remain_class))
    selected_id = []

    for target_class in seldom_class:
        idx_seldom_class.extend(sel_target_class(target_class, train_labels))
    for target_class in remain_class:
        idx_target_class = sel_target_class(target_class, train_labels)
        np.random.shuffle(idx_target_class)
        selected_id.extend(idx_target_class[0: sample_num_per_class])

        selected_id.extend(idx_seldom_class)
        x_all = train_data[selected_id]
        y_all = train_labels[selected_id]
        x_all = x_all.reshape((-1, input_shape[0], input_shape[1], input_shape[2]))
    return x_all, y_all