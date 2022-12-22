import os
import pandas as pd
import numpy as np

import copy
import time
import tensorflow as tf

# user define library
from utils.setup_NSL import NSLModel
from detector.detect import Detector


FEATURE_GROUP = [list(range(6)) + list(range(38, 121)), list(range(6, 18)), list(range(18, 27)), list(range(28, 37))]


class AEDetect:
    def __init__(self, args, data, features_num, train_data, train_labels, test_data, test_labels,
                 test_labels_one_hot, model):
        self.data = data
        self.args = args
        self.features_num = features_num
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_labels_one_hot = test_labels_one_hot

        self.x_all = np.concatenate(
            (train_data, test_data))  # concatenate the train and test data (for structure exploitation)
        test_labels_none = -1 * np.ones([test_labels.shape[0], ])  # the label of the test_data is set to -1
        self.y_all = np.concatenate((train_labels, test_labels_none))  # concatenate the train labels and -1 test labels

        self.adv_flags = []
        self.manifold = []
        self.uncertainty = []

        self.model = model
        self.detector = Detector(train_data, train_labels, features_num, model, max_drop_num=args.max_modify_num,
                                 drop_variants=args.modify_variants, modify_value=args.modify_value,
                                 threshold=0.01)

    def legistimated_mapping(self, input, adv, grad, scaler, max_v, min_v):
        adv = adv.reshape((1, -1))
        numerical_features_int_idx = list(range(20))
        numerical_features_int_idx.extend([27, 28])
        numerical_features_float_idx = list(range(20, 27))
        numerical_features_float_idx.extend(list(range(29, 37)))
        categorical_feature1_idx = list(range(37, 40))
        categorical_feature2_idx = list(range(40, 110))
        categorical_feature3_idx = list(range(110, 121))
        numerical_features = copy.deepcopy(adv[0, 0:37])
        numerical_features = np.reshape(numerical_features, (1, -1))
        numerical_features = scaler.inverse_transform(numerical_features)
        max_v_inverse = scaler.inverse_transform(max_v[0:37])
        if min_v:
            min_v_inverse = scaler.inverse_transform(min_v[0:37])
        else:
            min_v_inverse = np.zeros(121)  # scaler.inverse_transform(min_v[0:37])

        for idx in numerical_features_int_idx:
            if numerical_features[0, idx] < min_v_inverse[idx]:
                numerical_features[0, idx] = min_v_inverse[idx]
            elif numerical_features[0, idx] > max_v_inverse[idx]:
                numerical_features[0, idx] = max_v_inverse[idx]
            if idx <= 5:
                numerical_features[0, idx] = np.round(numerical_features[0, idx])
            else:
                if grad[0, idx] > 0:
                    numerical_features[0, idx] = np.maximum(min_v_inverse[idx], np.floor(numerical_features[0, idx]))
                elif grad[0, idx] < 0:
                    numerical_features[0, idx] = np.minimum(max_v_inverse[idx], np.floor(numerical_features[0, idx]) + 1)

        categorical_feature1 = copy.deepcopy(input[0, categorical_feature1_idx]).reshape((1, -1))
        categorical_feature2 = copy.deepcopy(input[0, categorical_feature2_idx]).reshape((1, -1))
        categorical_feature3 = copy.deepcopy(input[0, categorical_feature3_idx]).reshape((1, -1))

        # re-transform the numerical feature
        numerical_features = scaler.transform(numerical_features)
        mapped_features = np.concatenate(
            (numerical_features, categorical_feature1, categorical_feature2, categorical_feature3), axis=1)
        return mapped_features

    def run_detect_detail(self, samples, true_labels):
        """
        Adversarial Examples Crafting Target NN Model
        param  original_lab: original label of a data point in test set
        param target_lab:  # target label of an AE attack
        """
        detect_clean_samples = []
        labels = []
        mani_clf_difs = []
        adv_flag_ls = []
        start_time = time.time()
        for i in range(samples.shape[0]):
            print('>>>>>>>>>>> samples: {}/{}, time (seconds): {}'.format(i, len(samples), time.time() - start_time))
            pred = self.model(samples[i:i+1])
            pred = tf.nn.softmax(pred).eval()
            print("True Lab: {}, Pred: {} ".format(np.argmax(true_labels[i]), np.argmax(pred)))
            adv_flag, pred_labels, _, diff, manifold, mani_clf_dif = self.detector.detect(samples[i:i+1], pred)
            adv_flag_ls.append(adv_flag)
            if not np.isnan(mani_clf_dif):
                self.adv_flags.append(adv_flag)
                self.manifold.append(manifold)
                self.uncertainty.append(diff)
                mani_clf_difs.append(mani_clf_dif)
                print("Adv: {}, Ensemble Pred: {} ".format(adv_flag, pred_labels))
                if not adv_flag:
                    detect_clean_samples.append(samples[i])
                    labels.append(true_labels[i])

        # get samples that are detect as clean
        detect_clean_samples = np.array(detect_clean_samples)
        labels = np.array(labels)
        adv_flag_ls = np.array(adv_flag_ls)
        self.save_uncertainty()
        return adv_flag_ls, detect_clean_samples, labels, mani_clf_difs

    def run_detect_for_mixture(self, samples):
        """
        Adversarial Examples Crafting Target NN Model
        param  original_lab: original label of a data point in test set
        param target_lab:  # target label of an AE attack
        """
        print('evaluating samples scores.............')
        start_time = time.time()
        pred_adv = self.model(samples)
        pred_adv = tf.nn.softmax(pred_adv).eval()
        scores_1, scores_2 = self.detector.detect_para(samples, pred_adv)
        print('elapse time for {} samples is {} seconds'.format(len(scores_2), time.time() - start_time))
        return scores_1, scores_2

    def save_uncertainty(self):

        if self.uncertainty:
            uncertainty = {'uncertainty': self.uncertainty}
            uncertainty_df = pd.DataFrame(uncertainty)
            if not os.path.exists('result'):
                os.makedirs('result')
            fp = self.args.save_filepath + '_' + str(self.args.dataset) + '_' + str(self.args.attack) + '.csv'
            uncertainty_df.to_csv(fp)
