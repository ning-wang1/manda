from __future__ import division, absolute_import, print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist, cifar10
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.132, 'bim-a': 0.015, 'bim-b': 0.122},
    'nsl-kdd': {'fgsm-nsl': 0.2, 'bim-a-nsl': 0.14, 'cw-nsl': 0.2},
    'cicids': {'fgsm-cicids': 0.003, 'bim-a-cicids': 0.003, 'cw-cicids': 0.0006}
}
# Set random seed
np.random.seed(0)


def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x <= 0.5)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 0.5

    return np.reshape(x, original_shape)


def get_noisy_samples(data, X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    elif 'nsl-kdd' in dataset or 'cicids' in dataset:
        X_test_noisy = np.minimum(np.maximum(X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                                                       size=X_test.shape), data.min_v), data.max_v)
        # X_test_noisy = X_test
    else:
        warnings.warn("Using pre-set Gaussian scale sizes to craft noisy "
                      "samples. If you've altered the eps/eps-iter parameters "
                      "of the attacks used, you'll need to update these. In "
                      "the future, scale sizes will be inferred automatically "
                      "from the adversarial samples.")
        # Add Gaussian noise to the samples
        X_test_noisy = np.minimum(np.maximum(X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape), -0.5), 0.5)

    return X_test_noisy



# def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
#     """
#     TODO
#     :param model:
#     :param X:
#     :param nb_iter:
#     :param batch_size:
#     :return:
#     """
#     output_dim = model.layers[-1].output.shape[-1]
#     get_output = K.function(
#         [model.layers[0].input, K.learning_phase()],
#         [model.layers[-1].output]
#     )
#
#     def predict():
#         n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#         output = np.zeros(shape=(len(X), output_dim))
#         for i in range(n_batches):
#             output[i * batch_size:(i + 1) * batch_size] = \
#                 get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
#         return output
#
#     preds_mc = []
#     for i in tqdm(range(nb_iter)):
#         preds_mc.append(predict())
#
#     return np.asarray(preds_mc)
#
#
# def get_deep_representations(model, X, batch_size=256):
#     """
#     TODO
#     :param model:
#     :param X:
#     :param batch_size:
#     :return:
#     """
#     # last hidden layer is always at index -4
#     output_dim = model.layers[-4].output.shape[-1]
#     get_encoding = K.function(
#         [model.layers[0].input, K.learning_phase()],
#         [model.layers[-4].output]
#     )
#
#     n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#     output = np.zeros(shape=(len(X), output_dim))
#     for i in range(n_batches):
#         output[i * batch_size:(i + 1) * batch_size] = \
#             get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]
#
#     return output


def get_mc_predictions(model, dataset, X, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    if 'nsl-kdd' in dataset:
        output_dim = model.get_layer('dense_1').output.shape[-1]  # the last layer
        get_output = K.function(
            [model.get_layer('dense').input, K.learning_phase()],
            [model.get_layer('dense_1').output])     # first layer input and last layer output
    elif 'cicids' in dataset:
        output_dim = model.get_layer('dense_1').output.shape[-1]  # the last layer
        get_output = K.function(
            [model.get_layer('conv2d').input, K.learning_phase()],
            [model.get_layer('dense_1').output])  # first layer input and last layer output
    else:
        output_dim = model.get_layer('dense_2').output.shape[-1]
        get_output = K.function(
            [model.get_layer('conv2d').input, K.learning_phase()],
            [model.get_layer('dense_2').output])

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)


def get_deep_representations(model, dataset, X, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    if 'nsl-kdd' in dataset:
        output_dim = model.get_layer('dense').output.shape[-1]    # last hidden layer
        get_encoding = K.function(
            [model.get_layer('dense').input, K.learning_phase()],
            [model.get_layer('dense').output])     # first layer input and last hidden layer output
    elif 'cicids' in dataset:
        output_dim = model.get_layer('dense').output.shape[-1]    # last hidden layer
        get_encoding = K.function(
            [model.get_layer('conv2d').input, K.learning_phase()],
            [model.get_layer('dense').output])     # first layer input and last hidden layer output
    else:
        output_dim = model.get_layer('dense_1').output.shape[-1]    # last hidden layer
        get_encoding = K.function(
            [model.get_layer('conv2d').input, K.learning_phase()],
            [model.get_layer('dense_1').output])     # first layer input and last hidden layer output

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output


def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def get_value(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))
    return values, labels


def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score
