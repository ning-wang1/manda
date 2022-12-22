from __future__ import division, absolute_import, print_function
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
from utils.setup_mnist import MNIST, MNISTModel

dataset = 'mnist'
attack = 'cw'
change_threshold = 0.01
i = 1

advs = np.load('data/Adv_%s_%s_%s_%s.npy' % (dataset, attack, change_threshold, i))
X = np.load('data/clean_%s_%s_%s_%s.npy' % (dataset, attack, change_threshold, i))
Y = np.load('data/labels_%s_%s_%s_%s.npy' % (dataset, attack, change_threshold, i))

with tf.Session() as sess:
    model_dir = "models/mnist.h5"
    model = MNISTModel(model_dir, sess)
    y_preds = model.model(advs).eval()
    # print the generated adversarial examples
    # for i in range(len(advs)):

    i=3
    y_pred = y_preds[i]
    if not np.argmax(y_pred) == np.argmax(Y[i]):
        print('successful attack')
        print('true label of original image is: ', np.argmax(Y[i]))
        clean_img = np.squeeze(X[i])
        plt.imshow(clean_img, cmap='gray')
        plt.show()
        plt.imsave('figures/original.pdf', clean_img, cmap='gray')

        print('crafted image is predicted as: ', np.argmax(y_pred))
        adv_img = np.squeeze(advs[i])
        plt.imshow(adv_img, cmap='gray')
        plt.show()
        plt.imsave('figures/adv.pdf', adv_img, cmap='gray')
