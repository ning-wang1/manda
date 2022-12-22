## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import sys
import cv2
from imutils import paths
import pickle
from PIL import Image
import gzip
import urllib.request
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Input, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.semi_supervised import LabelSpreading
# from keras.utils import np_utils
from tensorflow.keras.models import load_model

import argparse
import tensorflow as tf

from utils.setup_mnist import MNISTModel, MNIST, MNISTAEModel
from utils.setup_cifar import CIFARModel, CIFAR, CIFARAEModel
from utils.setup_can import CAN, CANAEModel, CANModel
from utils.setup_cicids import CICIDS, CICIDSModel, CICIDSAEModel
import numpy as np
import math
import random
# from setup_inception import ImageNet, InceptionModel
from experiment_builder_cv import AEDetect
from sklearn import metrics
from utils.classifier import evaluate_sub, get_success_advs, random_select, compute_roc, compute_roc_1
from utils.classifier import get_correctly_pred_data

sys.path.append('../')
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def detect_detail(samples, labs, labs_ae_flag):
    # reject AEs inferred by detector, eval acc on the detect as clean data
    detect_as_clean_x, y, manifold_scores = AED.run_detect_detail(samples, labs)

    # eval detection performance
    evaluate_sub('Detection', labs_ae_flag, AED.adv_flags[-len(labs_ae_flag):])
    fpr, tpr, auc_score = compute_roc_1(labs_ae_flag, manifold_scores, plot=True)
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    # Print and save results
    ae_flags = labs_ae_flag.transpose()[0]
    manifold = np.array(AED.manifold)[-len(labs_ae_flag):]

    print("Only take a small sample from all the test records to show the detect detail")
    print('AE detected by manifold inconsistency for the sample: {}/{} '.format(np.sum(ae_flags * manifold),
                                                                                len(samples)))

    print('Clean data that result in False Positive: {}/{} '.format(np.sum((1 - ae_flags) * manifold),
                                                                    len(samples)))


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--save_filepath", type=str, default='result/uncertainty6.csv')
    parser.add_argument("--threshold", type=int, default=0.01)
    parser.add_argument("--modify_variants", type=int, default=3)
    parser.add_argument("--max_modify_num", type=int, default=200)
    parser.add_argument("--modify_value", type=int, default=0.015)  # mnist 0.3 # cicids 0.0006
    parser.add_argument("--attack", type=str, default='fgsm-cicids')
    parser.add_argument("--dataset", type=str, default='cicids')
    parser.add_argument('--change_threshold', type=float, default=0.01, help=' percentage of change in each feature')
    parser.add_argument('--intrusion_category', type=int, default=4)
    args = parser.parse_args()

    return args

def read_img(imagePaths):
    data =[]

    for (i, imagePath) in enumerate(imagePaths):
        # image = cv2.imread(imagePath)
        image = Image.open(imagePath)
        image = image.convert("RGB")
        image = image.resize([32, 32])
        image = np.asarray(image)/255
        data.append(image)
    return data


def get_label(imagePaths):
    label = []
    for (i, imagePath) in enumerate(imagePaths):
        if 'truck' in imagePath:
            label.append(2)
        elif 'train' in imagePath:
            label.append(0)
        else:
            label.append(1)
    return label

class My_Model(Model):
    def __init__(self):
        super(My_Model, self).__init__()
        params = [64, 64, 128, 64, 8, 128, 64, 64, 3]
        self.conv1 = Conv2D(params[0], 3, activation='relu')
        self.conv2 = Conv2D(params[1], 3, activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(params[2], 3, activation='relu')
        self.conv4 = Conv2D(params[3], 3, activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(params[4], activation='relu')
        self.d2 = Dense(params[5], activation='relu')
        self.d3 = Dense(params[6])

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


class CIFAR100:
    def __init__(self, type=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        file = '/home/ning/extens/GitHub/ae_detect/aeDetect/data/cifar-100-python/meta'
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        print(dict.keys())
        ls = dict[b'fine_label_names']
        for i, name in enumerate(ls):
            if name in [b'streetcar', b'tractor', b'train', b'pickup_truck']:
                print(f'{i} : {name}')
        train_idx = np.where(y_train==90)[0]
        car_idx = np.where(y_train==81)[0]
        truck_idx = np.where(y_train == 58)[0]
        x_train = x_train/255
        x_test = x_test/255

        imagepaths = list(paths.list_images('/home/ning/extens/GitHub/ae_detect/aeDetect/data/truck1/'))
        trucks = read_img(imagepaths)
        trucks = np.stack(trucks, axis=0)

        if type == 0:
            data_sel = x_train[train_idx, :, :, :]/ 255
            labels_sel = np.zeros(500).astype(int)
        elif type == 1:
            data_sel = x_train[car_idx, :, :, :] / 255
            labels_sel = np.ones(500).astype(int)
        elif type == 2:
            data_sel = x_train[truck_idx, :, :, :] / 255
            labels_sel = (2*np.ones(500)).astype(int)
        elif type==3:
            data_sel = np.concatenate((x_train[train_idx[0:200], :, :, :],
                                       x_train[car_idx, :, :, :],
                                       x_train[truck_idx, :, :, :]), axis=0)
            labels_sel = np.concatenate((np.zeros(200).astype(int),
                                         np.ones(500).astype(int),
                                         (2*np.ones(500)).astype(int)), axis=0)
        else:
            data_sel = np.concatenate((x_train[train_idx[0:200], :, :, :],
                                       x_train[car_idx, :, :, :],
                                       x_train[truck_idx, :, :, :],
                                       trucks[0:115, :, :, :]), axis=0)
            labels_sel = np.concatenate((np.zeros(200).astype(int),
                                         np.ones(500).astype(int),
                                         (2*np.ones(615)).astype(int)), axis=0)

        num = len(labels_sel)
        labels_sel_new = np.zeros((num,3)).astype(int)
        labels_sel_new[np.arange(num), labels_sel] = 1
        train_len = int(0.8 * num)
        idx = np.arange(num)
        np.random.shuffle(idx)

        self.train_data = data_sel[idx[0:train_len], :, :, :]
        self.train_labels = labels_sel_new[idx[0:train_len]]

        self.validation_data = data_sel[idx[train_len:], :, :, :]
        self.validation_labels = labels_sel_new[idx[train_len:]]

        train_idx = np.where(y_test == 90)[0]
        car_idx = np.where(y_test == 81)[0]
        truck_idx = np.where(y_test == 58)[0]

        if type==3:
            data_sel = np.concatenate((x_test[train_idx, :, :, :],
                                       x_test[car_idx, :, :, :],
                                       x_test[truck_idx, :, :, :]), axis=0)
            labels_sel = np.concatenate((np.zeros(100).astype(int),
                                         np.ones(100).astype(int),
                                         (2 * np.ones(100)).astype(int)), axis=0)
        else:
            data_sel = np.concatenate((x_test[train_idx, :, :, :],
                                       x_test[car_idx, :, :, :],
                                       x_test[truck_idx, :, :, :],
                                       trucks[0:100, :, :, :]), axis=0)
            labels_sel = np.concatenate((np.zeros(100).astype(int),
                                         np.ones(100).astype(int),
                                         (2 * np.ones(200)).astype(int)), axis=0)

        self.test_data = data_sel
        num = len(labels_sel)
        labels_sel_new = np.zeros((num, 3)).astype(int)
        labels_sel_new[np.arange(num), labels_sel] = 1
        self.test_labels = labels_sel_new


class COCO:
    def __init__(self, img_size=256):
        self.data_root = '/home/ning/extens/GitHub/ae_detect/aeDetect/data/COCO/'
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.train = datagen.flow_from_directory(self.data_root, target_size=(img_size, img_size),
                                                 batch_size=32, shuffle=True)
        self.test = datagen.flow_from_directory(self.data_root, target_size=(img_size, img_size),
                                                 batch_size=32, shuffle=True)

        num = 100
        data, labels = self.split_data(num)
        labels = np.argmax(labels, axis=1)
        train_len = int(0.7*num)
        val_len = int(0.8*num)

        self.train_data = data[0:train_len, :, :, :]
        self.train_labels = labels[0:train_len]

        self.validation_data = data[train_len:val_len, :, :, :]
        self.validation_labels = labels[train_len:val_len]

        self.test_data = data[val_len:, :, :, :]
        self.test_labels = labels[val_len:]

    def split_data(self, num=100):
        train_data = []
        train_labels = []
        for idx, (x, y) in enumerate(self.train):
            if idx < num:
                train_data.append(x)
                train_labels.append(y)
            else:
                return np.concatenate(train_data, axis=0), np.concatenate(train_labels, axis=0)

    def model_train(self):
        BATCH_SIZE = 32
        IMG_HEIGHT = 256
        IMG_WIDTH = 256
        ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                              input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        ResNet50.trainable = False
        net = tf.keras.models.Sequential()
        net.add(ResNet50)
        net.add(tf.keras.layers.GlobalAveragePooling2D())
        net.add(tf.keras.layers.Dense(3, activation='softmax'))
        net.summary()
        net.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        epoch_steps = len(self.train_data)
        val_steps = len(self.test_data)
        train_data, train_labels = self.split_data()
        net.fit(
            train_data, train_labels,
            steps_per_epoch=int(len(train_data)/BATCH_SIZE),
            epochs=3,
        )
        # net.fit_generator(
        #     self.train,
        #     steps_per_epoch=epoch_steps,
        #     epochs=3,
        #     validation_data=self.test,
        #     validation_steps=val_steps
        # )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    a = CIFAR100()
    # c = COCO()
    # c.model_train()
    # # x_train, y_train = c.split_data()
    # # print('completed')
    #
    # np.random.seed(1)
    # tf.random.set_seed(1)
    # args = get_args()
    # args.intrusion_category = 3
    #
    # data, model_dir = COCO(), 'models/demo.h5'
    ae_model_dir = "/home/ning/extens/GitHub/ae_detect/aeDetect/models/c100_ae.h5"
    #
    # load generated AEs
    imagepaths = list(paths.list_images('/home/ning/extens/GitHub/ae_detect/aeDetect/data/coco_ae/'))
    print(imagepaths)
    adv_samples = read_img(imagepaths)
    y_adv = get_label(imagepaths)
    # labels = (np.ones(len(adv_samples)) * 3).astype(int)
    # adv_samples = a.test_data

    data = a.train_data
    with tf.compat.v1.Session() as sess:
        encoder_model = CIFARAEModel(ae_model_dir, sess)
        outputs = encoder_model.get_latent(a.train_data)

        if not type(outputs) is np.ndarray:
            outputs = outputs.eval()

        consistency_model = LabelSpreading(gamma=6)
        consistency_model.fit(outputs, a.train_labels.argmax(1))
        correct = 0

        i = 0
        for x, y in zip(adv_samples, y_adv):
            i+=1
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            x_test_latent = encoder_model.get_latent(x)
            if not type(x_test_latent) is np.ndarray:
                x_test_latent = x_test_latent.eval()

            #  manifold consist with the clf output or not
            pred_consist = consistency_model.predict_proba(x_test_latent)  # the output of consistency model
            pred_label_consist = np.argmax(pred_consist)
            print('Manifold pred label: {}'.format(pred_label_consist))
            if y == pred_label_consist:
                correct += 1
            print(f'predicted label by model: {y}, idx {i}')
            # print('the classes list is: {}'.format(consist_model.classes_))
            print(pred_consist)
        print('correct data', correct)









# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# class ExperimentBuilder(object):
#     def __init__(self,train_data, test_data, epochs=10, lr=0.001):
#
#         self.model = My_Model()
#         self.train_data = train_data
#         self.test_data = test_data
#         self.epochs = epochs
#         self.lr = lr
#
#         self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         self.optimizer = tf.keras.optimizers.Adam()
#
#     @tf.function
#     def train_step(self, images, labels):
#         with tf.GradientTape() as tape:
#             # training=True is only needed if there are layers with different
#             # behavior during training versus inference (e.g. Dropout).
#             predictions = self.model(images, training=True)
#             loss = self.loss_object(labels, predictions)
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#         train_loss(loss)
#         train_accuracy(labels, predictions)
#
#
#     @tf.function
#     def test_step(self, images, labels):
#         # training=False is only needed if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         predictions = self.model(images, training=False)
#         t_loss = self.loss_object(labels, predictions)
#
#         test_loss(t_loss)
#         test_accuracy(labels, predictions)
#
#     def train(self):
#         """
#         Model Training
#         """
#         whole_len = len(self.train_data)
#
#         with tf.compat.v1.Session():
#             for epoch in range(self.epochs):
#                 # Reset the metrics at the start of the next epoch
#                 train_loss.reset_states()
#                 train_accuracy.reset_states()
#                 test_loss.reset_states()
#                 test_accuracy.reset_states()
#                 batch_idx = 0
#                 for inputs, targets in self.train_data:
#                     batch_idx += 1
#                     if batch_idx >= whole_len:
#                         break
#                     print(f'training batch: {batch_idx}/{whole_len}')
#                     # inputs, targets = inputs.to(self.device), targets.to(self.device)
#                     targets = np.argmax(targets, axis=1)
#                     self.train_step(inputs, targets)
#
#                     print(
#                         f'Epoch {epoch + 1}, '
#                         f'Loss: {train_loss.result().eval()}, '
#                         f'Accuracy: {train_accuracy.result().eval() * 100}, '
#                         f'Test Loss: {test_loss.result().eval()}, '
#                         f'Test Accuracy: {test_accuracy.result().eval() * 100}')

#
# class COCO_data():
#     def __init__(self):
#         self.train_data, self.train_labels, self.test_data, self.test_labels = self.read_data()
#
#     def read_data(self):
#         dirs =  ['car_data/', 'car_data/', 'car_data/']
#         dirs = [r'/home/ning/extens/GitHub/ae_detect/aeDetect/data/COCO/' + i for i in dirs]
#         imagePaths_ls = [list(paths.list_images(d)) for d in dirs]
#
#         num = 100
#         data = []
#         labels = []
#         for l, imagePaths in enumerate(imagePaths_ls):
#             label = l
#             for (i, imagePath) in enumerate(imagePaths[:num]):
#                 image = cv2.imread(imagePath)
#                 data.append(image)
#                 labels.append(label)
#
#         length = len(labels)
#         idx = np.arange(length)
#         np.random.shuffle(idx)
#         train_len = int(0.7 * length)
#         train_idx = list(idx[: train_len])
#         data = np.array(data)
#         train_data = data[train_idx, :, :]
#         train_labels = np.array(labels[idx[: train_len]])
#
#         test_data = np.array(data[idx[train_len:]])
#         test_labels = np.array(labels[idx[train_len:]])
#
#         return train_data, train_labels, test_data, test_labels