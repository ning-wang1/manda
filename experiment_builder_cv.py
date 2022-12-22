import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random

import tensorflow as tf

# user define library
from attacks.attack_l2 import CarliniL2
from utils.setup_mnist import MNISTModel
from utils.setup_cifar import CIFARModel
from detector.detect_cv import Detector
import warnings
import time


class AEDetect:
    def __init__(self, args, data, model, encoder_model):
        self.args = args
        self.data = data
        self.input_shape = [data.train_data.shape[1], data.train_data.shape[2], data.train_data.shape[3]]

        self.x_all = np.concatenate((data.train_data, data.validation_data, data.test_data))
        test_labels_none = -1 * np.ones([data.test_labels.shape[0], ])  # the label of the test_data is set to -1
        train_labels_numerical = np.argmax(data.train_labels, axis=1)
        validation_labels_numerical = np.argmax(data.validation_labels, axis=1)
        self.y_all = np.concatenate((train_labels_numerical, validation_labels_numerical, test_labels_none))

        self.adv_flags = []
        self.manifold = []
        self.uncertainty = []

        image_size = data.train_data.shape[1]
        num_channels = data.train_data.shape[3]

        if args.dataset is 'cicids':
            class_n = 13
        elif 'cicids_binary' in args.dataset:
            class_n = 2
        else:
            class_n = 10

        self.detector = Detector(data, [image_size, image_size, num_channels], model, encoder_model, class_num=class_n,
                                 max_modify_num=args.max_modify_num,
                                 modify_value=args.modify_value,
                                 modify_variants=args.modify_variants, threshold=args.threshold)
        self.model = model

    def plot_setting(self):
        warnings.filterwarnings('ignore')
        pd.set_option('display.max_columns', None)
        # np.set_printoptions(threshold=np.nan)
        np.set_printoptions(precision=3)
        sns.set(style="darkgrid")
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

    def run_detect_detail(self, samples, true_labels):
        detect_clean_samples = []
        labels = []
        mani_clf_difs = []
        start_time = time.time()
        for i in range(len(samples)):
            print('>>>>>>>>>>> samples: {}/{}, time (seconds): {}'.format(i, len(samples), time.time()-start_time))
            pred_adv = self.model(samples[i:i + 1])
            pred_adv = tf.nn.softmax(pred_adv).eval()
            pred_adv_lab = np.argmax(pred_adv)
            print("True Lab: {}, Pred: {} ".format(np.argmax(true_labels[i]), pred_adv_lab))
            adv_flag, pred_labels, final_label, diff, manifold,  mani_clf_dif = \
                self.detector.detect(samples[i], pred_adv, self.data.feature_mean)
            print("Adv: {}, Ensemble preds: {}, Final result: {} \n".format(adv_flag, pred_labels, final_label))

            self.adv_flags.append(adv_flag)
            self.manifold.append(manifold)
            self.uncertainty.append(diff)
            mani_clf_difs.append(mani_clf_dif)

            if not adv_flag:
                detect_clean_samples.append(samples[i])
                labels.append(true_labels[i])

        # get samples that are detect as clean
        detect_clean_samples = np.array(detect_clean_samples)
        labels = np.array(labels)

        self.save_uncertainty()
        return detect_clean_samples, labels, mani_clf_difs

    def run_detect_for_mixture(self, samples, dataset):
        print('evaluating samples scores.............')
        start_time = time.time()
        pred_adv = self.model(samples)
        pred_adv = tf.nn.softmax(pred_adv).eval()
        scores_1, scores_2 = self.detector.detect_para(samples, pred_adv, dataset, self.data.feature_mean)
        print('elapse time for {} samples is {} seconds'.format(len(scores_2), time.time()-start_time))
        return scores_1, scores_2

    def save_uncertainty(self):
        if self.uncertainty:
            uncertainty = {'uncertainty': self.uncertainty}
            uncertainty_df = pd.DataFrame(uncertainty)
            if not os.path.exists('result'):
                os.makedirs('result')
            uncertainty_df.to_csv(self.args.save_filepath)
