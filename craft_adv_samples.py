from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
# import keras.backend as K
# import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model
import sys
import random
import copy
from matplotlib import pyplot as plt
from utils.setup_NSL import NSL_KDD, NSLModel
from utils.setup_can import CAN, CANModel
from attacks.attack_l2 import CarliniL2
from attacks.attack_l0 import CarliniL0
from attacks.attack_l2_restrict import CarliniL2Res
from attacks.attack_l2_restrict_2d import  CarliniL2_res_2d
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
# ATTACK_PARAMS = {
#     'mnist': {'eps': 0.300, 'eps_iter': 0.010},
#     'cifar': {'eps': 0.050, 'eps_iter': 0.005},
#     'svhn': {'eps': 0.130, 'eps_iter': 0.010},
#     'nsl-kdd': {'eps': 0.83, 'eps_iter': 0.02},
#     'CAN_DoS': {'eps': 0.300, 'eps_iter': 0.010},
#     'cicids_binary': {'eps': 0.080, 'eps_iter': 0.030},
#     'cicids': {'eps': 0.08, 'eps_iter': 0.030}
# }

ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'nsl-kdd': {'eps': 0.83, 'eps_iter': 0.02},
    'CAN_DoS': {'eps': 0.300, 'eps_iter': 0.010},
    'cicids_binary': {'eps': 0.080, 'eps_iter': 0.030},
    'cicids': {'eps': 0.08, 'eps_iter': 0.030}
}

L2 = 40
FRAME_SIZE = 28
DIM = 28


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

    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=-0.5,
            clip_max=0.5, batch_size=batch_size
        )
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=-0.5,
            clip_max=0.5, batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=-0.5, clip_max=0.5
        )
    elif attack == 'cw':
        # CW attack
        X_adv, X, Y, _ = generate_ae_cw(args, sess, args.craft_num, model, X, Y)

    elif attack == 'fgsm-nsl' or attack == 'fgsm-cicids':
        # FGSM restricted attack
        print('Crafting restricted fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method_restricted(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            modified_percentage=args.change_threshold,
            clip_min=data.min_v, clip_max=data.max_v, batch_size=batch_size,
            dataset=dataset
        )

    elif attack in ['bim-a-nsl', 'bim-b-nsl', 'bim-a-cicids', 'bim-b-cicids']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method_restricted(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], modified_percentage=args.change_threshold,
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=data.min_v,
            clip_max=data.max_v, batch_size=batch_size, dataset=args.dataset
        )
        if 'bim-a' in attack:
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif 'cw-cicids' in attack:
        X_adv, X, Y, _ = generate_ae_cw_cicids(sess, args.craft_num, model, X, Y, args.change_threshold,
                                               start_0=args.i, binary=args.binary,
                                               intrusion_category=args.intrusion_category)
    else:
        X_adv, X, Y, _ = generate_ae_cw_nsl(args.i, sess, args.craft_num, model, X, Y, args.change_threshold)

    # L2 distance
    l2_diff = np.linalg.norm(X_adv.reshape((len(X_adv), -1)) - X.reshape((len(X_adv), -1)), axis=1).mean()
    print("Average L-2 perturbation size of the adv test set: {}".format(l2_diff))
    # save l-2 distance to file
    with open('data/stat_%s_%s_%s_%s.txt' % (args.dataset, args.attack, args.change_threshold, args.i), 'w') as f:
        print("Average L-2 perturbation size of the adv test set: {}".format(l2_diff), file=f)

    # save AEs, true labels and the corresponding clean data of Adv
    np.save('data/crafted_ae/Adv_%s_%s_%s_%s.npy' % (args.dataset, args.attack, args.change_threshold, args.i), X_adv)
    np.save('data/crafted_ae/clean_%s_%s_%s_%s.npy' % (args.dataset, args.attack, args.change_threshold, args.i), X)
    np.save('data/crafted_ae/labels_%s_%s_%s_%s.npy' % (args.dataset, args.attack, args.change_threshold, args.i), Y)
    return X_adv, X, Y


def plot_img(img, image_size):
    img=img.reshape(( image_size,  image_size))
    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.show()
    plt.close()


def plot_ids(frame_1, frame_2, frame_size, idx):
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, frame_size)
    frame_1 = (frame_1 * 2).astype(int)
    frame_2 = (frame_2 * 2).astype(int)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    change_num = 0
    for i in range(frame_size):
        ax.text(0.5, i, str(frame_1[i, :]), ha='center', va='center', color='black')
        if np.all(frame_1[i] == frame_2[i]):
            ax.text(1.5, i, str(frame_2[i, :]), ha='center', va='center', color='black')
        else:
            change_num += 1
            ax.text(1.5, i, str(frame_2[i, :]), ha='center', va='center', color='red')
    # ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize='small')
    if change_num < 11:
        print('change number', change_num)
        fig.savefig('result/figures/{}.pdf'.format(idx), bbox_inches='tight', dpi=100)


def to_can_problem_space(adv, img, max_v=0.5, min_v=0, random_mode=False, proportion=0.9):
    adv_1d = adv.flatten()
    img_1d = img.flatten()
    adv = list(adv_1d)
    img = list(img_1d)
    mid = (max_v + min_v)/2
    threshold = 0.25

    index = np.arange(len(adv))
    change_loc = index[np.abs(img_1d - adv_1d) < .0001]

    if random_mode:
        indices = copy.deepcopy(change_loc)
        np.random.shuffle(indices)
        num = int(len(change_loc) * proportion)
        selected_ind = indices[0:num]
        adv = [adv[i] if i in selected_ind else img[i] for i in range(len(adv))]

        # new_adv = [adv[i] if i in loc else max_v-adv[i] for i in range(len(adv))]
    new_adv = list(map(lambda x: max_v if x > threshold else min_v, adv))
    problem_space_adv = np.array(new_adv).reshape([FRAME_SIZE, DIM])
    return problem_space_adv


def generate_ae_cw(args, sess, samples_num, model, X, Y):
    """
    generate AEs or a collection of clean data, depends on args.attack_flag
    :return: samples, true_labels and target_labels
    """
    def generate_data(X, Y, samples, targeted=False, start=0, inception=False):
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
            if targeted:
                if inception:
                    seq = random.sample(range(1, 1001), 10)
                else:
                    seq = range(Y.shape[1])

                for j in seq:
                    if (j == np.argmax(Y[start + i])) and (inception == False):
                        continue
                    inputs.append(X[start + i])
                    targets.append(np.eye(Y.shape[1])[j])
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
    inputs_to_generate_ae, true_labels, target_labels = generate_data(X, Y, samples_num,
                                                                      targeted=True, start=0, inception=False)
    if 'CAN_DoS' in args.dataset:
        model_to_attack = CANModel("models/DoS.h5", sess)
    elif 'CAN_Fuzzy' in args.dataset:
        model_to_attack = CANModel("models/Fuzzy.h5", sess)
    elif 'CAN_gear' in args.dataset:
        model_to_attack = CANModel( "models/gear.h5", sess)
    elif 'CAN_RPM' in args.dataset:
        model_to_attack = CANModel("models/RPM.h5", sess)
    elif 'cicids' in args.dataset:
        model_to_attack = CICIDSModel('models/cicids.h5', sess)
    else:
        model_to_attack = MNISTModel("models/mnist.h5", sess)

    attack = CarliniL2(sess, model_to_attack, batch_size=9, max_iterations=1000, confidence=0)
    # attack = CarliniL0(sess, model_to_attack, max_iterations=500, largest_const=1e5)
    samples = attack.attack(inputs_to_generate_ae, target_labels)  # generate ae

    return samples, inputs_to_generate_ae, true_labels, target_labels


def generate_ae_cw_nsl(batch_idx, sess, samples_num, model, X, Y, change_threshold):

    def generate_data(X, Y, samples):
        idxs = np.array([i for i in range(Y.shape[0])])
        # np.random.shuffle(idxs)
        # idx_sel = idxs[0:samples]
        idx_sel = idxs[batch_idx*samples: (batch_idx+1)*samples]
        inputs = X[idx_sel].reshape([samples, -1])
        original_labels = Y[idx_sel].reshape([samples, -1])
        targets = 1-original_labels

        return inputs, targets, original_labels

    samples = []
    x = []
    true_labels = []
    target_labels = []

    data = NSL_KDD([('DoS', 0.0)])
    print('Input Shape-- features: {}, num: {}'.format(X.shape[1], X.shape[0]))
    inputs, targets, original_labels = generate_data(X, Y, samples=samples_num)
    start_time = time.time()
    for i in range(samples_num):
        print('GENERATE AE {}/{}, time elapse: {} '.format(i, samples_num, time.time()-start_time))
        input_x = inputs[i].reshape((1, -1))
        target = targets[i].reshape((1, -1))

        boxmin = np.maximum(data.min_v, input_x - (data.max_v - data.min_v) * 0.001)
        boxmax = np.minimum(data.max_v, input_x + (data.max_v - data.min_v) * change_threshold)

        model_to_attack = NSLModel("models/nsl_kdd_Dos.h5", data.train_data.shape[1], sess)
        attack = CarliniL2Res(sess, model_to_attack, boxmin, boxmax, batch_size=1, max_iterations=1000)
        adv, grad, obest_l2 = attack.attack(input_x, target)  # generate ae for the selected test data point
        # adv = legistimated_mapping(input, adv, grad, data.scaler, data.max_v, data.min_v)

        if obest_l2 < L2:
            samples.append(adv)
            x.append(inputs[i])
            true_labels.append(original_labels[i])
            target_labels.append(targets[i])

    true_labels = np.array(true_labels)
    target_labels = np.array(target_labels)
    samples = np.array(samples).reshape((-1, data.train_data.shape[1]))
    x = np.array(x).reshape((-1, data.train_data.shape[1]))
    print('successful attack: {}'.format(len(samples)))

    return samples, x, true_labels, target_labels


def generate_ae_cw_cicids(sess, samples_num, model, X, Y, change_threshold, start_0,
                          binary=False, intrusion_category=4):
    """
        generate AEs or a collection of clean data, depends on args.attack_flag
        :return: samples, true_labels and target_labels
        """

    def generate_data(X, Y, samples_num, targeted=True, start=0, inception=False):
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
        start = 100 * start
        n = 0
        i = 0
        while n < samples_num and i < len(Y):
            true_label = np.argmax(Y[start + i])
            if targeted:
                # the target is 0, i.e., benign
                if not true_label == 0:
                    inputs.append(X[start + i])
                    targets.append(np.eye(Y.shape[1])[0])
                    original_lab.append(Y[start + i])
                    n += 1
            else:
                inputs.append(X[start + i])
                targets.append(Y[start + i])
            i += 1

        inputs = np.array(inputs)
        targets = np.array(targets)
        if original_lab:
            original_lab = np.array(original_lab)

        return inputs, original_lab, targets

    shape = (X.shape[1], X.shape[2], X.shape[3])

    print('Shape of Train Data: {}\n'.format(shape))
    # generate data for crafting AEs
    inputs_to_generate_ae, true_labels, target_labels = generate_data(X, Y, samples_num,
                                                                      targeted=True, start=start_0, inception=False)
    samples = []

    if binary:
        data = CICIDS(attack_cat=4)
        model_to_attack = CICIDSModel('models/cicids_binary.h5', sess, binary=True)
    else:
        data = CICIDS()
        model_to_attack = CICIDSModel('models/cicids.h5', sess)

    start_time = time.time()
    for i in range(samples_num):
        print('GENERATE AE {}/{}, time elapse: {} '.format(i, samples_num, time.time()-start_time))
        input_x = inputs_to_generate_ae[i].reshape((1, 9, 9, 1))

        boxmin = np.maximum(data.min_v, input_x - (data.max_v - data.min_v) * 0.001)
        boxmax = np.minimum(data.max_v, input_x + (data.max_v - data.min_v) * change_threshold)

        attack = CarliniL2_res_2d(sess, model_to_attack,  batch_size=1, max_iterations=1000,
                                  confidence=0, boxmin=boxmin, boxmax=boxmax)
        target = target_labels[i:i+1]
        adv = attack.attack(input_x, target)  # generate ae

        samples.append(adv)

    samples = np.array(samples).reshape((-1, 9, 9, 1))
    print('successful attack: {}'.format(len(samples)))

    return samples, inputs_to_generate_ae, true_labels, target_labels


def main(args):
    with tf.Session() as sess:
        if 'mnist' in args.dataset:
            data, model_dir = MNIST(), "models/mnist.h5"
            model = MNISTModel(model_dir, sess)
        elif 'kdd' in args.dataset:
            data, model_dir = NSL_KDD([('DoS', 0.0)]), "models/nsl_kdd_Dos.h5"
            model = NSLModel(model_dir, data.train_data.shape[1], sess)
        elif args.dataset is 'cicids':
            data, model_dir = CICIDS(), 'models/cicids.h5'
            model = CICIDSModel(model_dir, sess)
        elif 'cicids_binary' in args.dataset:
            data, model_dir = CICIDS(attack_cat=args.intrusion_category), 'models/cicids_binary.h5'
            model = CICIDSModel(model_dir, sess, binary=True)

        X_test, Y_test = data.test_data, data.test_labels
        acc = model.evaluate(X_test, Y_test)
        print("Accuracy on the test set: %0.2f%%" % (100*acc))
        total_length = len(X_test)
        # plot_img(X_test[6])

        X_test, Y_test, x_remain, y_remain, _= get_correctly_pred_data(model.model, X_test, Y_test)
        print("the total number of test data points: {}; the number of malicious data points: {}".format(
            total_length, len(X_test)))
        # Craft one specific attack type
        advs, X, Y = craft_one_type(sess, model.model, args, X_test, Y_test, data,
                                    args.dataset, args.attack, args.batch_size)

        success = 0
        if 'CAN' in args.dataset:
            for i in range(len(advs)):
                adv = advs[i, :, :, 0]
                img = X[i, :, :, 0]
                adv_new = to_can_problem_space(adv, img)
                advs[i, :, :, 0] = adv_new
                if not np.argmax(model.model(advs[i:i+1]).eval) == np.argmax(Y[i]):
                    print('success')
                    success += 1
                    # plot_img(advs[i], image_size=DIM)
                    # plot_img(X[i], image_size=DIM)
                    plot_ids(X[i, :, :, 0], advs[i, :, :, 0], frame_size=DIM, idx=i)
        else:
            predicted = model.model(advs).eval()
            success = Y.shape[0] - np.count_nonzero(predicted.argmax(1) == Y.argmax(1))

        acc_adv = model.evaluate(advs, Y)

        if 'cw' in args.attack:
            total_num_adv = args.craft_num
        else:
            total_num_adv = len(X)

        print('attack success rate {}/{}'.format(success, total_num_adv))

        with open('data/crafted_ae/stat_%s_%s_%s_%s.txt' % (args.dataset, args.attack, args.change_threshold, args.i), 'a') as f:
            print('perturbation limitation in each feature {}'.format(args.change_threshold), file=f)
            print('attack success rate {}/{}'.format(success, total_num_adv), file=f)
            print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc_adv), file=f)


def combine_files(args, binary=False):
    """
    combine the separate files of AEs for CW attack. And get the attack success rate for an AE attack.
    :param args: the arguments (see get_args)
    """
    # load the separate files of AEs, and combine the AEs to a single array
    for i in range(1, 6):
        adv_file = 'data/crafted_ae/Adv_{}_{}_{}_{}.npy'.format(args.dataset, args.attack, args.change_threshold, i)
        labels_file = 'data/crafted_ae/labels_{}_{}_{}_{}.npy'.format(args.dataset, args.attack, args.change_threshold,
                                                                      i)
        clean_data_fie = 'data/crafted_ae/clean_{}_{}_{}_{}.npy'.format(args.dataset, args.attack,
                                                                        args.change_threshold, i)

        # load data
        x = np.load(adv_file)
        y = np.load(labels_file)
        x_clean = np.load(clean_data_fie)

        # combine data
        if i == 1:
            samples = x
            true_labels = y
            clean_samples = x_clean
        else:
            samples = np.concatenate((samples, x), axis=0)
            true_labels = np.concatenate((true_labels, y), axis=0)
            clean_samples = np.concatenate((clean_samples, x_clean), axis=0)

        # remove loaded data file
        # os.remove(adv_file)
        # os.remove(labels_file)
        # os.remove(clean_data_fie)
        # os.remove('data/stat_%s_%s_%s_%s.txt' % (args.dataset, args.attack, args.change_threshold, i))

    # save the combined AEs to a new file
    np.save('data/crafted_ae/Adv_%s_%s_%s_combined.npy' % (args.dataset, args.attack, args.change_threshold), samples)
    np.save('data/crafted_ae/labels_%s_%s_%s_combined.npy' % (args.dataset, args.attack, args.change_threshold), true_labels)
    np.save('data/crafted_ae/clean_%s_%s_%s_combined.npy' % (args.dataset, args.attack, args.change_threshold), clean_samples)

    # evaluate the overall ASR
    with tf.compat.v1.Session() as sess:
        if args.dataset is 'cicids':
            data, model_dir = CICIDS(), 'models/cicids.h5'
            model = CICIDSModel(model_dir, sess)
        elif 'cicids_binary' in args.dataset:
            data, model_dir = CICIDS(attack_cat=args.intrusion_category), 'models/cicids_binary.h5'
            model = CICIDSModel(model_dir, sess, binary=True)

        preds = model.predict(samples).eval()
    total_num = true_labels.shape[0]
    success = total_num - np.count_nonzero(preds.argmax(1) == true_labels.argmax(1))

    # save the ASR to file
    with open('data/stat_%s_%s_%s_combined.txt' % (args.dataset, args.attack, args.change_threshold), 'w') as f:
        print('attack success rate is: {}/{}'.format(success, total_num), file=f)


def evaluate_acc(args):
    """ evaluate the model accuracy on test set"""

    with tf.compat.v1.Session() as sess:
        if 'mnist' in args.dataset:
            data, model_dir = MNIST(), "models/mnist.h5"
            model = MNISTModel(model_dir, sess)
        elif 'kdd' in args.dataset:
            data, model_dir = NSL_KDD([('DoS', 0.0)]), "models/nsl_kdd_Dos.h5"
            model = NSLModel(model_dir, data.train_data.shape[1], sess)
        elif args.dataset is 'cicids':
            data, model_dir = CICIDS(), 'models/cicids.h5'
            model = CICIDSModel(model_dir, sess)
        elif 'cicids_binary' in args.dataset:
            data, model_dir = CICIDS(attack_cat=args.intrusion_category), 'models/cicids_binary.h5'
            model = CICIDSModel(model_dir, sess, binary=True)

        X_test, Y_test = data.test_data, data.test_labels
        acc = model.evaluate(X_test, Y_test)
        print("Accuracy on the test set: %0.2f%%" % (100*acc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='nsl-kdd',
                        help='possible: cicids, mnist, nsl_kdd, CAN_DoS,CAN_Fuzzy, CAN_gear, CAN_RPM')
    parser.add_argument('--attack', type=str, default='fgsm-nsl', help='possible:cw, fgsm-nsl, jsma, bim-a-nsl, bim-b-nsl, cw-nsl')
    parser.add_argument('--craft_num', type=int, default=50, help='the num to craft, only applicable to cw and cw-nsl')
    parser.add_argument('--change_threshold', type=float, default=0.1, help=' percentage of change in each feature')
    parser.add_argument('--i', type=str, default='1')
    parser.add_argument('--binary', default=False, action='store_true')
    parser.add_argument('--intrusion_category', type=int, default=4)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    np.random.seed(1)
    args = get_args()
    args.dataset = 'nsl-kdd'
    args.attack = 'fgsm-nsl'
    args.change_threshold = 0.1
    args.intrusion_category = 10  # only active for cicids dataset
    args.binary = True
    args.i = 1
    main(args)
    # combine_files(args)
    # evaluate_acc(args)
