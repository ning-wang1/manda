from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import sys
from matplotlib import pyplot as plt
from utils.setup_NSL import NSL_KDD, NSLModel
from attacks.attack_l2_restrict import CarliniL2Res
from utils.setup_mnist import MNIST, MNISTModel
from utils.classifier import get_correctly_pred_data
from attacks.attacks_restricted import (fast_gradient_sign_method_restricted, basic_iterative_method_restricted)
from attacks.attack_l2_restrict import transform_loc_to_idx
import copy
import time
from matplotlib import pyplot as plt

sys.path.append("..")
# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'nsl-kdd': {'eps': 0.83, 'eps_iter': 0.02},
}
L2 = 1000
intrusions = {'Dos': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}
numerical_features_int_idx = list(range(20))
numerical_features_int_idx.extend([27, 28])

numerical_features_float_idx = list(range(20, 27))
numerical_features_float_idx.extend(list(range(29, 37)))


# modified_fea = [1, 5, 9, 10, 11, 13, 16, 17, 18, 19] + list(range(23, 42))
# modified_fea = [1, 5, 10, 11, 13, 16, 17, 18, 19] + [23, 24] + [32, 33]
# modified_fea = [1, 5, 10, 11, 13, 18] + [23, 24] + [32, 33]
modified_fea = [1, 5] + [23, 24] + [32, 33]
modified_fea = transform_loc_to_idx(modified_fea)


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
            # BIM-A: For each sample, select the time step where that sample first became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B: For each sample, select the very last time step
            X_adv = results[-1]
    else:
        X_adv, X, Y, _ = generate_ae_cw_nsl(sess, args, args.craft_num, args.start, model,
                                            X, Y, args.change_threshold)

    # L2 distance
    l2_diff = np.linalg.norm(X_adv.reshape((len(X_adv), -1)) - X.reshape((len(X_adv), -1)), axis=1).mean()
    print("Average L-2 perturbation size of the adv test set: {}".format(l2_diff))

    return X_adv, Y


def generate_ae_cw_nsl(sess, args, samples_num, start, model, X, Y, change_threshold):

    def generate_data(X, Y, samples, start):
        idxs = np.array([i for i in range(Y.shape[0])])
        np.random.shuffle(idxs)
        idx_sel = idxs[start: start+samples]
        inputs = X[idx_sel].reshape([samples, -1])
        original_labels = Y[idx_sel].reshape([samples, -1])
        targets = 1-original_labels

        return inputs, targets, original_labels

    def generate_data_in_order(X, Y, samples, start):
        inputs = X[start: start+samples].reshape([samples, -1])
        original_labels = Y[start: start+samples].reshape([samples, -1])
        targets = 1 - original_labels

        return inputs, targets, original_labels

    samples = []
    x = []
    true_labels = []
    target_labels = []
    num_success_problem_adv_mal = 0
    num_success_problem_adv_ben = 0

    data = NSL_KDD([(args.intrusion_name, intrusions[args.intrusion_name])])
    print('Input Shape-- features: {}, num: {}'.format(X.shape[1], X.shape[0]))
    inputs, targets, original_labels = generate_data(X, Y, samples=samples_num, start=start)
    # inputs, targets, original_labels = generate_data_in_order(X, Y, samples=samples_num, start=start)
    start_time = time.time()

    for i in range(samples_num):
        print('GENERATE AE {}/{}, time elapse: {} '.format(i, samples_num, time.time()-start_time))
        input_x = inputs[i].reshape((1, -1))
        target = targets[i].reshape((1, -1))
        boxmin = np.maximum(data.min_v, input_x - (data.max_v - data.min_v) * change_threshold)
        boxmax = np.minimum(data.max_v, input_x + (data.max_v - data.min_v) * change_threshold)
        # boxmin = np.maximum(data.min_v, input_x - (input_x - data.min_v) * change_threshold)
        # boxmax = np.minimum(data.max_v, input_x + (input_x - data.min_v) * change_threshold)

        model_to_attack = NSLModel("models/nsl_kdd_" + args.intrusion_name + ".h5", data.train_data.shape[1], sess)
        attack = CarliniL2Res(sess, model_to_attack, boxmin, boxmax, batch_size=1, max_iterations=1000)
        adv, grad, obest_l2 = attack.attack(input_x, target)  # generate ae for the selected test data point

        if obest_l2 < L2:
            samples.append(adv)
            x.append(inputs[i])
            true_labels.append(original_labels[i])
            target_labels.append(targets[i])

            x_problem = reverse_fea(input_x, data.scaler)
            adv_legit, adv_problem = legitimated_mapping(input_x, adv, x_problem, data.scaler)
            adv_legit = adv

            np.set_printoptions(precision=2)
            print('The successful generation-----------------------------------------')
            print('input in feature space:\n {} '.format(input_x))
            print('adv in feature space:\n {} '.format(adv))
            print('legit adv in feature space:\n {} '.format(adv_legit))
            print('adv in problem space:\n {} '.format(adv_problem))

            print('prediction on input', np.argmax(model(input_x).eval()))
            print('prediction on adv', np.argmax(model(adv).eval()))
            print('prediction on legitimated adv', np.argmax(model(adv_legit).eval()))

            pred_input = np.argmax(model(input_x).eval())
            pred_adv = np.argmax(model(adv_legit).eval())

            if not pred_adv == pred_input:
                if pred_input == 0:
                    num_success_problem_adv_mal += 1
                    if num_success_problem_adv_mal == 1:
                        up_mal, down_mal = find_dif(x_problem, adv_problem)
                    else:
                        up_the_input, down_the_input = find_dif(x_problem, adv_problem)
                        up_mal += up_the_input
                        down_mal += down_the_input
                else:
                    num_success_problem_adv_ben += 1
                    if num_success_problem_adv_ben == 1:
                        up_ben, down_ben = find_dif(x_problem, adv_problem)
                    else:
                        up_the_input, down_the_input = find_dif(x_problem, adv_problem)
                        up_ben += up_the_input
                        down_ben += down_the_input

    true_labels = np.array(true_labels)
    target_labels = np.array(target_labels)
    samples = np.array(samples).reshape((-1, data.train_data.shape[1]))
    x = np.array(x).reshape((-1, data.train_data.shape[1]))

    print('successful feature-space attack: {}'.format(len(samples)))
    print('successful problem-space attack: {}'.format(num_success_problem_adv_mal + num_success_problem_adv_ben))
    if num_success_problem_adv_mal > 0:
        print('up features for malicious input, ', up_mal)
        print('down features for malicious input', down_mal)
    if num_success_problem_adv_ben > 0:
        print('up features for benign input, ', up_ben)
        print('down features for benign input', down_ben)
    return samples, x, true_labels, target_labels


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


def legitimated_mapping(x, adv, x_problem_space, scaler):
    """ mapping the feature space adv to the problem space"""

    # reverse the numerical feature of crafted adv
    adv = adv.reshape((1, -1))
    numerical_fea = copy.deepcopy(adv[0, 0:37])
    numerical_fea = np.reshape(numerical_fea, (1, -1))
    numerical_fea = scaler.inverse_transform(numerical_fea)

    # # force int feature to int
    for idx in numerical_features_int_idx:
        numerical_fea[0, idx] = np.round(numerical_fea[0, idx])
    for idx in numerical_features_float_idx:
        numerical_fea[0, idx] = np.round(numerical_fea[0, idx], 5)

    # nullify the change on un_modified features
    unmodified_fea = list(set(list(range(37))) - set(modified_fea))
    numerical_fea[0, unmodified_fea] = x_problem_space[0, unmodified_fea]

    # form the problem space adv using the reverse features
    problem_space = np.concatenate((numerical_fea, x_problem_space[:, -3:]), axis=1)

    # re-transform the problem space adv into feature space
    numerical_fea = scaler.transform(numerical_fea)
    mapped_features = np.concatenate((numerical_fea, x[:, 37:]), axis=1)

    return mapped_features, problem_space


def reverse_fea(x, scaler):
    """transform feature space data to problem space"""
    # numerical features
    numerical_fea = copy.deepcopy(x[0, 0:37])
    numerical_fea = np.reshape(numerical_fea, (1, -1))

    # reverse numerical features
    reverse_num = scaler.inverse_transform(numerical_fea)
    for idx in numerical_features_int_idx:
        reverse_num[0, idx] = np.round(reverse_num[0, idx])
    for idx in numerical_features_float_idx:
        reverse_num[0, idx] = np.round(reverse_num[0, idx], 5)

    # categorical features: from one-hot to indices
    categorical_feature1 = copy.deepcopy(x[0, 37:40]).reshape((1, -1))
    categorical_feature2 = copy.deepcopy(x[0, 40: 110]).reshape((1, -1))
    categorical_feature3 = copy.deepcopy(x[0, 110: 121]).reshape((1, -1))
    reverse_cat1 = np.argmax(categorical_feature1)
    reverse_cat2 = np.argmax(categorical_feature2)
    reverse_cat3 = np.argmax(categorical_feature3)

    # build problem space adv using the reverse features
    problem_space = np.concatenate((reverse_num, np.array([[reverse_cat1]]), np.array([[reverse_cat2]]),
                                    np.array([[reverse_cat3]])), axis=1)

    return problem_space


def main(args):
    with tf.compat.v1.Session() as sess:
        # data, model_dir = NSL_KDD([('DoS', 0.0)]), "models/nsl_kdd_Dos.h5"
        data = NSL_KDD([(args.intrusion_name, intrusions[args.intrusion_name])])
        model_dir = 'models/nsl_kdd_' + args.intrusion_name + '.h5'
        model = NSLModel(model_dir, data.train_data.shape[1], sess)

        X_test, Y_test = data.test_data, data.test_labels

        acc = model.evaluate(X_test, Y_test)
        print("Accuracy on the test set: %0.2f%%" % (100 * acc))

        X_test, Y_test, x_remain, y_remain, _ = get_correctly_pred_data(model.model, X_test, Y_test, target=0)

        # Craft one specific attack type
        adv, Y = craft_one_type(sess, model.model, args, X_test, Y_test, data, args.dataset,
                                args.attack, args.batch_size)
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
    parser.add_argument('--craft_num', type=int, default=150, help='the num to craft, only applicable to cw and cw-nsl')
    parser.add_argument('--start', type=int, default=0, help='start idx for crafting, applicable to cw and cw-nsl')
    parser.add_argument('--change_threshold', type=float, default=0.1, help=' only applicable to cw-nsl')
    parser.add_argument('--i', type=str, default='1')
    parser.add_argument('--intrusion_name', type=str, default='U2R')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    args.dataset = ['nsl_kdd']
    args.attack = ['cw-nsl']
    args.intrusion_name = 'Dos'
    args.start = 0
    main(args)
    # data = NSL_KDD([(args.intrusion_name, intrusions[args.intrusion_name])])
    # data.get_feature_mean()
