# load controller area network (CAN) bus data and build model for CAN bus data.


import numpy as np
import os
import sys
import pandas as pd
import time
import pickle
import gzip
import urllib.request

import tensorflow as tf
from tensorflow.keras.models import Model as K_Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import load_model

sys.path.append('../')
BATCH_SIZE = 28000
FRAME_SIZE = 28
dim = 28


def myhex2bin(x_arr, y_arr, data):
    # extract the ID column from data, and transform the hex format to binary.
    # Attach pre-fix to it to build image-like data
    arr_len = len(x_arr)
    batches = int(arr_len/BATCH_SIZE)
    x_frame = []
    y_frame = []
    time_start = time.time()
    # To speed up, process it in batches
    for b in range(batches):
        print('processing {}/{}, time elapse {}'.format((b+1)*BATCH_SIZE, arr_len, time.time()-time_start))
        x_batch = x_arr[b*BATCH_SIZE: (b+1)*BATCH_SIZE]
        y_batch = y_arr[b*BATCH_SIZE: (b+1)*BATCH_SIZE]
        x_new = np.zeros((int(BATCH_SIZE / FRAME_SIZE), FRAME_SIZE, dim))
        y_new = np.zeros(int(BATCH_SIZE / FRAME_SIZE))
        for idx, x in enumerate(x_batch):
            x = bin(int(x, 16))
            x = x.replace('0b', '')
            x_split = [int(i) for i in x]
            length = len(x_split)
            x_pre = [0 for i in range(dim-length)]
            x = x_pre + x_split
            x_frame.append(x)
            y_frame.append(y_batch[idx])

            if (idx+1)%FRAME_SIZE==0:
                frame_n = int(idx/FRAME_SIZE)
                x_new[frame_n, :, :] = np.array(x_frame).reshape((FRAME_SIZE, dim))

                # deal with nan in label y
                if np.nan in y_frame:
                    loc = y_frame.index(np.nan)
                    y_frame[loc] = data.iloc[b*BATCH_SIZE + frame_n*FRAME_SIZE + loc, 5]

                # if there is a 'T' in the frame, label it as attack
                y_new[frame_n] = 'T' in y_frame
                x_frame = []
                y_frame = []
        x_new.astype(int)
        y_new.astype(int)
        if b==0:
            x_arr_new = x_new
            y_arr_new = y_new
        else:
            x_arr_new = np.concatenate((x_arr_new, x_new), axis=0)
            y_arr_new = np.concatenate((y_arr_new, y_new), axis=0)

    return x_arr_new, y_arr_new


def can_pre_process_2_classes():
    # build a training dataset for a binary classifier. group every attack type with the normal type
    m_folder = 'data/car_hacking/'
    file_names = ['DoS_dataset.csv', 'Fuzzy_dataset.csv', 'gear_dataset.csv',
                  'RPM_dataset.csv']
    normal_file_name = 'normal_run_data.txt'

    # read intrusion data
    for i, file_name in enumerate(file_names):
        print('loading.......', file_name)
        file_path = m_folder + file_name
        data = pd.read_csv(file_path)
        id = data.iloc[:, 1]
        y = np.array(data.iloc[:, 11])
        data_x, data_y = myhex2bin(id, y, data)

        print('The shape of {} data is {}'.format(file_name, data_x.shape))

        # divide the data into three sets: train, validation, and test.
        total_num = len(data_x)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        TRAIN_SIZE = 0.72
        VALIDATION_SIZE = 0.08
        TEST_SIZE = 0.20

        train_idx = indices[0: int(TRAIN_SIZE * total_num)]
        valid_idx = indices[int(TRAIN_SIZE * total_num): int((VALIDATION_SIZE + TRAIN_SIZE) * total_num)]
        test_idx = indices[int((VALIDATION_SIZE + TRAIN_SIZE) * total_num): -1]

        train_data = data_x[train_idx, :]
        train_labels = data_y[train_idx]
        test_data = data_x[test_idx, :]
        test_labels = data_y[test_idx]
        validation_data = data_x[valid_idx, :]
        validation_labels = data_y[valid_idx]

        # saving the data to file
        attack_type = file_name.split('_')[0]
        if not os.path.exists(('data/car_hacking/'+attack_type)):
            os.makedirs('data/car_hacking/'+attack_type)
        np.save('data/car_hacking/' + attack_type + '/train_data.npy', train_data)
        np.save('data/car_hacking/' + attack_type + '/train_labels.npy', train_labels)
        np.save('data/car_hacking/' + attack_type + '/test_data.npy', test_data)
        np.save('data/car_hacking/' + attack_type + '/test_labels.npy', test_labels)
        np.save('data/car_hacking/' + attack_type + '/validation_data.npy', validation_data)
        np.save('data/car_hacking/' + attack_type + '/validation_labels.npy', validation_labels)


class CAN:
    def __init__(self, attack_type='DoS'):
        # possible attack_types: DoS, Fuzzy, RPM, gear
        labels = np.array([0, 1])

        data_scale = 0.5
        test_num = 5000
        # all the data is 0 or 1, transform the data to -0.5 and 0.5
        train_data = np.load('data/car_hacking/' + attack_type + '/train_data.npy') * data_scale
        train_labels = np.load('data/car_hacking/' + attack_type + '/train_labels.npy')
        train_labels = train_labels.reshape((-1, 1))==labels
        self.train_data = train_data.reshape((-1, FRAME_SIZE, dim, 1))
        self.train_labels = train_labels.astype(int)

        test_data = np.load('data/car_hacking/' + attack_type + '/test_data.npy')* data_scale
        test_data = test_data.reshape((-1, FRAME_SIZE, dim, 1))
        test_labels = np.load('data/car_hacking/' + attack_type + '/test_labels.npy')
        test_labels = test_labels.reshape((-1, 1))==labels
        test_labels = test_labels.astype(int)

        # self.test_data = test_data
        # self.test_labels = test_labels
        self.test_data = test_data[0:test_num, :, :, :]
        self.test_labels = test_labels[0:test_num]

        validation_data = np.load('data/car_hacking/' + attack_type + '/validation_data.npy')* data_scale
        validation_labels = np.load('data/car_hacking/' + attack_type + '/validation_labels.npy')
        validation_labels = validation_labels.reshape((-1, 1))==labels
        self.validation_data = validation_data.reshape((-1, FRAME_SIZE, dim, 1))
        self.validation_labels = validation_labels.astype(int)


class CANModel:
    def __init__(self, restore, session=None):
        params = [64, 64, 128, 128, 256, 256, 2]
        self.num_channels = 1
        self.image_size = dim
        self.num_labels = 2

        model = Sequential()

        model.add(Conv2D(params[0], (3, 3),
                         input_shape=(self.image_size, self.image_size, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[1], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(params[2], (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[3], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(params[4]))
        model.add(Activation('relu'))
        model.add(Dense(params[5]))
        model.add(Activation('relu'))
        model.add(Dense(params[6]))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

    def evaluate(self, x, y):
        # loss, acc = self.model.evaluate(x, y, batch_size=128)
        # print("test loss {}, test acc: {}".format(loss, acc))
        # x=x[0:1000, :, :,:]
        # y = y[0:1000]
        predicted = self.model(x).eval()
        acc = np.count_nonzero(predicted.argmax(1) == y.argmax(1)) / y.shape[0]

        return acc


class CANAEModel:
    def __init__(self, restore, session=None):
        params = [64, 64, 128, 128, 49, 7, 128, 128, 64, 64, 1]
        inp = Input((dim, dim, 1))
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
        model = K_Model(inp, decoded)
        encoder = K_Model(inp, l1)
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











# def myhex2bin_1(data):
#     data_len = len(data)
#     batches = int(data_len/BATCH_SIZE)
#     x_ls = []
#     for b in range(batches):
#         print('processing {}/{}'.format((b+1)*BATCH_SIZE, data_len))
#         x_batch = data.iloc[b*BATCH_SIZE: (b+1)*BATCH_SIZE, 1]
#         x_new = np.zeros((BATCH_SIZE, 29))
#         for idx, x in enumerate(x_batch):
#             x = bin(int(x, 16))
#             x = x.replace('0b', '')
#             x_split = [int(i) for i in x]
#             length = len(x_split)
#             x_pre = [0 for i in range(29-length)]
#             x = x_pre + x_split
#             x_new[idx, :] = np.array(x)
#             x_new.astype(int)
#         x_ls.append(x_new)
#     return x_ls
#
#
# def can_pre_process():
#     if not os.path.exists("../data/car_hacking"):
#         print('dataset not found in folder /data/can_hacking')
#     m_folder = 'data/car_hacking/'
#     file_names = ['DoS_dataset.csv', 'Fuzzy_dataset.csv', 'gear_dataset.csv',
#                   'RPM_dataset.csv', 'normal_run_data.txt']
#
#     for i, file_name in enumerate(file_names):
#         print('loading.......', file_name)
#         file_path = m_folder + file_name
#         if 'csv' in file_name:
#             data = pd.read_csv(file_path)
#             id = data.iloc[:, 1]
#         else:
#             with open(file_path, "r") as text_file:
#                 data = text_file.readlines()
#             id = [record.replace(' ', '')[31:35] for record in data]
#
#         x = myhex2bin(id)
#         y = [i] * len(x)
#         print('The shape of {} data is {}'.format(file_name, x.shape))
#         if i == 0:
#             data_x = x
#             data_y = y
#         else:
#             data_x = np.concatenate((data_x, x))
#             data_y.extend(y)
#
#     total_num = len(data_x)
#     indices = np.arange(total_num)
#     np.random.shuffle(indices)
#
#     TRAIN_SIZE = 0.72
#     VALIDATION_SIZE = 0.08
#     TEST_SIZE = 0.20
#
#     train_idx = indices[0: int(TRAIN_SIZE * total_num)]
#     valid_idx = indices[int(TRAIN_SIZE * total_num): int((VALIDATION_SIZE + TRAIN_SIZE) * total_num)]
#     test_idx = indices[int((VALIDATION_SIZE + TRAIN_SIZE) * total_num): -1]
#
#     data_x = np.array(data_x)
#     data_y = np.array(data_y)
#     train_data = data_x[train_idx, :]
#     train_labels = data_y[train_idx]
#     test_data = data_x[test_idx, :]
#     test_labels = data_y[test_idx]
#     validation_data = data_x[valid_idx, :]
#     validation_labels = data_y[valid_idx]
#
#     np.save('data/car_hacking/train_data.npy', train_data)
#     np.save('data/car_hacking/train_labels.npy', train_labels)
#     np.save('data/car_hacking/test_data.npy', test_data)
#     np.save('data/car_hacking/test_labels.npy', test_labels)
#     np.save('data/car_hacking/validation_data.npy', validation_data)
#     np.save('data/car_hacking/validation_labels.npy', validation_labels)