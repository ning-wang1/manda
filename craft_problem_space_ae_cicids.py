from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf

import sys
import random
import copy
from matplotlib import pyplot as plt

from attacks.attack_l2_restrict_2d import CarliniL2_res_2d
from utils.setup_mnist import MNIST, MNISTModel
from utils.setup_cicids import CICIDSAEModel, CICIDS, CICIDSModel
from utils.setup_cifar import CIFAR, CIFARModel
from utils.classifier import get_correctly_pred_data
from attacks.attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method)
from attacks.attacks_restricted import (fast_gradient_sign_method_restricted, basic_iterative_method_restricted,
                            saliency_map_method)

import time

sys.path.append("..")
# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'nsl-kdd': {'eps': 0.83, 'eps_iter': 0.02},
}
L2 = 1000

modified_fea = list(range(1, 30)) + list(range(34, 43)) + list(range(51, 56)) + list(range(62, 66))
numerical_int_feas = list(range(1, 8)) + \
                     [10, 11, 18, 19, 20, 23, 24, 25, 28, 29, 34, 35, 38, 39, 55, 62, 63, 64, 65]


def craft_one_type(sess, model, args, X, Y, data, dataset, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """

    if attack == 'fgsm-nsl':
        # FGSM restricted attack
        print('Crafting restricted fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method_restricted(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], modified_percentage=args.change_threshold,
            clip_min=data.min_v,
            clip_max=data.max_v, batch_size=batch_size
        )
    elif attack in ['bim-a-nsl', 'bim-b-nsl']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method_restricted(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], modified_percentage=args.change_threshold,
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=data.min_v,
            clip_max=data.max_v, batch_size=batch_size
        )
        if attack == 'bim-a-nsl':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    else:
        X_adv, X, Y, _ = generate_ae_cicids(args, sess, args.craft_num, model, X, Y, data.scaler)

    # L2 distance
    l2_diff = np.linalg.norm(X_adv.reshape((len(X_adv), -1)) - X.reshape((len(X_adv), -1)), axis=1).mean()
    print("Average L-2 perturbation size of the adv test set: {}".format(l2_diff))

    return X_adv, Y


def generate_ae_cicids(args, sess, samples_num, model, X, Y, scaler):
    """
    generate AEs or a collection of clean data, depends on args.attack_flag
    :return: samples, true_labels and target_labels
    """
    def generate_data(X, Y, samples, targeted=True, start=0, inception=False):
        """
        Generate the input data to the attack algorithm.
        data: the images to attack
        samples: number of samples to use
        targeted: if true, construct targeted attacks, otherwise untargeted attacks
        start: offset into data to use
        inception: if targeted and inception, randomly sample 100 targets intead of 1000
        """
        inputs = []
        targets = []
        original_lab = []

        for i in range(samples):
            true_label = np.argmax(Y[start + i])
            if targeted:
                # the target is 0, i.e., benign
                if not true_label==0:
                    inputs.append(X[start + i])
                    targets.append(np.eye(Y.shape[1])[0])
                    original_lab.append(Y[start + i])
            else:
                inputs.append(X[start + i])
                targets.append(Y[start + i])

        inputs = np.array(inputs)
        targets = np.array(targets)
        if original_lab:
            original_lab = np.array(original_lab)

        return inputs, original_lab, targets

    shape = (X.shape[1], X.shape[2], X.shape[3])

    print('Shape of Train Data: {}\n'.format(shape))
    # generate data for crafting AEs
    inputs_to_generate_ae, true_labels, target_labels = generate_data(X, Y, samples_num,
                                                                      targeted=True, start=0, inception=False)
    model_to_attack = CICIDSModel('models/cicids.h5', sess)

    # launch attack and obtain AE samples
    attack = CarliniL2_res_2d(sess, model_to_attack, batch_size=1, max_iterations=1000, confidence=0)
    samples = attack.attack(inputs_to_generate_ae, target_labels)  # generate ae

    # initialize dictionary to save the statistics
    categories = [str(x) for x in range(13)]
    success_num_dict = dict()
    up_dict = dict()
    down_dict = dict()
    for c in categories:
        up_dict[c] = np.zeros([1, 81])
        down_dict[c] = np.zeros([1, 81])
        success_num_dict[c] = 0

    for i in range(len(samples)):
        input_x = inputs_to_generate_ae[i].reshape((1, 9, 9, 1))
        adv = samples[i]
        adv = adv.reshape((1, 9, 9, 1))
        adv_legit, adv_problem, x_problem = legitimated_mapping(input_x, adv, scaler)
        adv_legit = adv_legit.reshape((1, 9, 9, 1))

        # display the AE
        np.set_printoptions(precision=2)
        print('The successful generation-----------------------------------------')
        print('input in feature space:\n {} '.format(input_x))
        print('adv in feature space:\n {} '.format(adv))
        print('legit adv in feature space:\n {} '.format(adv_legit))
        print('adv in problem space:\n {} '.format(adv_problem))
        print('input in problem space:\n {}'.format(x_problem))

        # model prediction on original data and AE
        pred_input = np.argmax(model(input_x).eval())
        pred_adv = np.argmax(model(adv).eval())
        pred_adv_legit = np.argmax(model(adv_legit).eval())
        category = np.argmax(true_labels[i])

        print('prediction on input', pred_input)
        print('prediction on adv', pred_adv)
        print('prediction on legitimated adv', pred_adv_legit)

        # a successful AE
        if not pred_adv_legit == pred_input:
            success_num_dict[str(category)] += 1
            up_the_input, down_the_input = find_dif(x_problem, adv_problem)
            up_dict[str(category)] += up_the_input
            down_dict[str(category)] += down_the_input

    print('successful feature-space attack: {}'.format(len(samples)))
    print('successful problem-space attack: {}'.format(success_num_dict))

    for c in categories:
        print('Category {} (predicted as Benign), success num: {}'.format(c, success_num_dict[str(c)]))
        print('up features, ', up_dict[str(c)])
        print('down features', down_dict[str(c)])

    return samples, inputs_to_generate_ae, true_labels, target_labels


def find_dif(x, x_adv):
    up = np.zeros([1, x.shape[1]])
    down = np.zeros([1, x.shape[1]])
    dif = x_adv - x
    up = up + (dif > 0)
    down = down + (dif < 0)
    mask = np.zeros([1, x.shape[1]])
    mask[0, modified_fea] = 1
    up = up * mask
    down = down * mask
    return up, down


def legitimated_mapping(x, adv, scaler, max_v=0.5, min_v=-0.5):
    scale = 0.4
    adv = adv.reshape((1, -1)) / scale

    # reverse the crafted adv from (0,1) to original range
    inversed_feas = scaler.inverse_transform(adv[0, :78].reshape((1, -1)))
    adv_problem_space = np.concatenate((inversed_feas, np.zeros((1, 3))), axis=1)

    for idx in numerical_int_feas:
        adv_problem_space[0, idx] = np.round(adv_problem_space[0, idx])

    # reverse the input x (in the range (0,1)) to original range
    x = x.reshape((1, -1)) / scale
    inversed_feas = scaler.inverse_transform(x[0, :78].reshape((1, -1)))
    x_problem_space = np.concatenate((inversed_feas, np.zeros((1, 3))), axis=1)

    # nullify the change on un_modified features
    unmodified_fea = list(set(list(range(81))) - set(modified_fea))
    adv_problem_space[0, unmodified_fea] = x_problem_space[0, unmodified_fea]

    # re-transform the problem space adv into feature space
    feas = scaler.transform(adv_problem_space[0, :78].reshape((1, -1)))
    feas_last_3 = adv[0, -3:].reshape((1, -1))
    legistimate_adv_feature_space = np.concatenate((feas, feas_last_3), axis=1) * scale

    return legistimate_adv_feature_space, adv_problem_space, x_problem_space


def main(args):
    with tf.compat.v1.Session() as sess:
        if 'mnist' in args.dataset:
            data, model_dir = MNIST(), "models/mnist.h5"
            model = MNISTModel(model_dir, sess)
        else:
            # data, model_dir = NSL_KDD([('DoS', 0.0)]), "models/nsl_kdd_Dos.h5"
            data = CICIDS()
            model_dir = 'models/cicids.h5'
            model = CICIDSModel(model_dir, sess)

        # model accuracy on original test data
        X_test, Y_test = data.test_data, data.test_labels
        acc = model.evaluate(X_test, Y_test)
        print("Accuracy on the test set: %0.2f%%" % (100 * acc))
        # plot_img(X_test[6])

        # select a subset of test data that model gives correct predictions
        X_test, Y_test, x_remain, y_remain, _= get_correctly_pred_data(model.model, X_test, Y_test, target_not=0)

        # Craft one specific attack type
        adv, Y = craft_one_type(sess, model.model, args, X_test, Y_test, data,
                                args.dataset, args.attack, args.batch_size)
        # plot_img(adv[6])
        acc_adv = model.evaluate(adv, Y)

        print("Accuracy on the test set: %0.2f%%" % (100 * acc))
        print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc_adv))
        print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc_adv*acc))


def plot_fea_change(up, down):
    up_new = up[0, modified_fea].reshape(1, -1)
    down_new = down[0, modified_fea].reshape(1, -1)
    plt.figure()
    plt.bar(up_new)
    plt.bar(down_new)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='mnist', help='possible: mnist, nsl_kdd')
    parser.add_argument('--attack', type=str, default='cw', help='possible:cw, fgsm, jsma, bim-a, bim-b, cw-nsl')
    parser.add_argument('--craft_num', type=int, default=50, help='the num to craft, only applicable to cw and cw-nsl')
    parser.add_argument('--start', type=int, default=0, help='start idx for crafting, applicable to cw and cw-nsl')
    parser.add_argument('--change_threshold', type=float, default=0.1, help=' only applicable to cw-nsl')
    parser.add_argument('--i', type=str, default='1')
    parser.add_argument('--intrusion_name', type=str, default='U2R')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    args.dataset = 'cicids'
    args.attack = 'cw'
    args.start = 0
    main(args)
