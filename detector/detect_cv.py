import numpy as np
import random
from sklearn.semi_supervised import LabelSpreading
from scipy.special import comb
from scipy import stats
import tensorflow as tf

from utils.setup_mnist import MNISTModel, MNISTAEModel
from utils.setup_cifar import CIFARModel, CIFARAEModel
modified_fea = list(range(1, 30)) + list(range(34, 43)) + list(range(51, 56)) + list(range(62, 66))


class Detector:
    def __init__(self, data, input_shape, model, encoder_model, class_num,
                 max_modify_num, modify_value,  modify_variants, threshold):
        self.data = data
        self.max_modify_num = max_modify_num
        self.input_shape = input_shape
        self.features_num = 1
        for i in input_shape:
            self.features_num *= i
        possible_modify_variants = self.choice_num(self.features_num, max_modify_num)
        self.modify_variants = modify_variants
        self.threshold = threshold
        self.sigma = modify_value
        # self.uncertain_eval_type = 'consist'
        self.uncertain_eval_type = 'clf'

        self.encoder_model = encoder_model
        self.model = model

        self.consistency_model = {}
        x = np.concatenate((data.train_data, data.validation_data))
        y = np.concatenate((data.train_labels, data.validation_labels))
        y = np.argmax(y, axis=1)
        print('get maniflod.........(take a while)')
        #
        if class_num == 2:
            model = self.train_consistency_model_balanced(x, y, binary=True)
        elif class_num == 13:
            model = self.train_consistency_model_balanced(x, y)
        else:
            model = self.train_consistency_model(x, y, class_num)

        self.consistency_model['multi'] = model

        # for i in range(class_num):
        #     for j in range(i+1, class_num):
        #         model = self.train_consistency_model_balanced(x, y, i, j)
        #         self.consistency_model['{}-{}'.format(i, j)] = model

    def choice_num(self, n, m):
        """
        Computing the choice num by selecting m nodes or less than m nodes from n nodes
        """
        num = 0
        if m > n:
            print('value error')
        else:
            for i in range(m):
                num += comb(n, i + 1)
        return num

    def dropout_features(self, x, drop_ls, max_modify_num, drop_num=False):
        n = x.shape[0]
        x = x.reshape(n, -1)
        if not drop_num:
            drop_num = random.randrange(1, max_modify_num)
        drop_idx = random.sample(drop_ls, drop_num)
        selected = [xx for xx in range(self.features_num) if xx not in drop_idx]
        remain_features_num = len(selected)
        x_new = np.zeros((x.shape[0], remain_features_num))
        for i in range(x.shape[0]):
            x_new[i, :] = x[i, selected]
        return x_new, drop_num

    def modify_features(self, x, modify_ls, max_modify_num, modify_num=False):
        clip_min = self.data.min_v
        clip_max = self.data.max_v
        x = x.reshape(1, -1)
        x_new = np.zeros((x.shape[0], x.shape[1]))

        if not modify_num:
            modify_num = random.randrange(1, x.shape[1])
        modified_idx = random.sample(modify_ls, modify_num)
        un_modified_idx = [xx for xx in range(self.features_num) if xx not in modified_idx]
        x_new[0, modified_idx] = np.clip(x[0, modified_idx] + np.random.normal(0, self.sigma), clip_min, clip_max)
        x_new[0, un_modified_idx] = x[0, un_modified_idx]
        x_new = x_new.reshape((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return x_new

    def replace_features(self, x, feature_mean, predict):
        """ get noisy variants, method 1, design for cicids
        by replacing a subset of features as the average value of these features """
        data_num = x.shape[0]
        x_flatten = x.reshape(data_num, 81)
        th = -0.001

        noise = np.random.normal(loc=0, scale=self.sigma, size=x_flatten.shape)
        sel_pos = (noise > th).astype(int)
        # nullify the features that are in the sel_pos (by replace the feature with the average value of this feature)
        # x_noisy = x_flatten * sel_pos + (1-sel_pos) * feature_mean
        x_noisy = x_flatten * sel_pos + (1-sel_pos) * np.dot(predict, feature_mean)

        for i in range(self.modify_variants - 1):
            noise = np.random.normal(loc=0, scale=self.sigma, size=x_flatten.shape)
            sel_pos = (noise > th).astype(int)
            # x_new = x_flatten * sel_pos + (1-sel_pos) * feature_mean
            x_new = x_flatten * sel_pos + (1 - sel_pos) * np.dot(predict, feature_mean)
            x_noisy = np.concatenate((x_noisy, x_new), 0)
        x_noisy = x_noisy.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return x_noisy

    def add_noise(self, x):
        """ get noisy variants, method 2, design for cicids
        by purely adding noise"""
        clip_min = self.data.min_v
        clip_max = self.data.max_v

        data_num = x.shape[0]
        x_flatten = x.reshape(data_num, 81)
        mask = [1.0 if i in modified_fea else 0.0 for i in range(81)]
        # mask = [1.0 for i in range(81)]
        noise = np.random.normal(loc=0, scale=self.sigma, size=x_flatten.shape) * np.array(mask)
        x_noisy = np.minimum(np.maximum(x_flatten + noise, clip_min), clip_max)
        for i in range(self.modify_variants - 1):
            noise = np.random.normal(loc=0, scale=self.sigma, size=x_flatten.shape) * np.array(mask)
            x_new = np.minimum(np.maximum(x_flatten + noise, clip_min), clip_max)
            x_noisy = np.concatenate((x_noisy, x_new), 0)
        x_noisy = x_noisy.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return x_noisy

    def detect_para(self, x_test, nn_predict, dataset, feature_mean):
        """
        detect the adversarial example by evaluating model output uncertainty
        """
        x = x_test.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        consist_model = self.consistency_model['multi']

        x_test_latent = self.encoder_model.get_latent(x)
        if not type(x_test_latent) is np.ndarray:
            x_test_latent = x_test_latent.eval()

        #  step-1 manifold consist with the clf output or not
        pred_consist = consist_model.predict_proba(x_test_latent)         # the output of consistency model
        preds = np.concatenate((pred_consist, nn_predict), 0)
        preds = preds.reshape((2, len(pred_consist), -1))
        preds = np.swapaxes(preds, 0, 1)
        mani_clf_dif = self.uncertainty_1(preds)

        # step 2
        # for image input
        if 'mnist' in dataset:
            scale = 0.5
            x_noisy = np.minimum(np.maximum(x + np.random.normal(loc=0, scale=self.sigma, size=x.shape), -scale), scale)
            for i in range(self.modify_variants -1):
                x_new = np.minimum(np.maximum(x + np.random.normal(loc=0, scale=self.sigma, size=x.shape), -scale), scale)
                x_noisy = np.concatenate((x_noisy, x_new), 0)
        elif 'cicids' in dataset:
            # x_noisy = self.replace_features(x, feature_mean, nn_predict)
            x_noisy = self.add_noise(x)
        elif 'CAN' in dataset:
            x_noisy = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
            x_to_change = (x_noisy > 1).astype(int)
            x_noisy = x + x_to_change * ((x == 0).astype(int) * 0.5 - (x > 0).astype(int) * 0.5)
            for i in range(self.modify_variants - 1):
                x_new = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
                x_to_change = (x_new > 0.6).astype(int)
                x_new = x + x_to_change * ((x == 0).astype(int) * 0.5 - (x > 0).astype(int) * 0.5)
                x_noisy = np.concatenate((x_noisy, x_new), 0)

        y_pred, _ = self.clf_output(x_noisy)
        y_pred = np.concatenate((y_pred, nn_predict), 0)
        y_pred = y_pred.reshape((self.modify_variants+1, len(x), y_pred.shape[1]))
        # y_pred = y_pred.reshape((self.modify_variants, len(x), y_pred.shape[1]))
        y_pred = np.swapaxes(y_pred, 0, 1)

        diff = self.uncertainty_1(y_pred)

        return mani_clf_dif, diff

    def detect(self, x_test, nn_predict, feature_mean):
        """
        detect the adversarial example by evaluating model output uncertainty
        """
        diff = None
        manifold_inconsis_num = 0
        y_pred_ls = []
        pred_labels = []

        modified_fea = [x for x in range(self.features_num)]
        x = x_test.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        consist_model = self.consistency_model['multi']

        x_test_latent = self.encoder_model.get_latent(x)
        if not type(x_test_latent) is np.ndarray:
            x_test_latent = x_test_latent.eval()

        #  manifold consist with the clf output or not
        pred_consist = consist_model.predict_proba(x_test_latent)         # the output of consistency model
        pred_label_consist = np.argmax(pred_consist)
        print('Manifold pred label: {}'.format(pred_label_consist))
        # print('the classes list is: {}'.format(consist_model.classes_))
        print(pred_consist)
        print(nn_predict)
        mani_clf_dif = self.uncertainty([pred_consist, nn_predict])

        if not pred_label_consist == np.argmax(nn_predict):
            adv_flag = True
            manifold_inconsis_num += 1
            final_label = pred_label_consist
            print('---------MANIFOLD INCONSISTENT-------')
            # if pred_label_consist < np.argmax(nn_predict):
            #     model_name = str(pred_label_consist) + '-' + str(np.argmax(nn_predict))
            # else:
            #     model_name = str(np.argmax(nn_predict)) + '-' + str(pred_label_consist)
            # manifold_2_classes = self.consistency_model[model_name]
            # pred = manifold_2_classes.predict(x_test_latent)[0]
            # print('Manifold between 2 classes, mani: {}, nn: {}'.format(pred, np.argmax(nn_predict)))
            # if pred == np.argmax(nn_predict):
            #     adv_flag = False
            #     final_label = pred
            # else:
            #     adv_flag = True
            #     manifold_inconsis_num += 1
            #     final_label = pred_label_consist
            #     print('---------MANIFOLD INCONSISTENT-------')
        else:
            # x_new = np.zeros((self.modify_variants, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            # for i in range(self.modify_variants):
            #     x_new[i] = self.modify_features(x[0], modified_fea, self.max_modify_num)

            x_new = self.replace_features(x, feature_mean, nn_predict)
            # x_new = self.add_noise(x)
            y_pred, pred_labels = self.clf_output(x_new)
            y_pred_ls = list(y_pred)
            pred_labels = list(pred_labels)

            if self.uncertain_eval_type is 'consist':
                pred_labels.append(pred_label_consist)
                y_pred_ls.append(np.ravel(pred_consist))
            else:
                pred_labels.append(np.argmax(nn_predict))
                y_pred_ls.append(np.ravel(nn_predict))

            y_pred_ls_new = y_pred_ls
            diff = self.uncertainty(y_pred_ls_new)
            final_label = stats.mode(pred_labels)[0][0]

            if diff > self.threshold:
                adv_flag = True
            else:
                adv_flag = False
            print('PRED on origianl input : {}, Uncertainty： {} \n'.format(np.argmax(nn_predict), diff))
        return adv_flag, pred_labels, final_label, diff, manifold_inconsis_num, mani_clf_dif

    def clf_output(self, x):
        outputs = self.model(x)
        outputs = tf.nn.softmax(outputs).eval()
        pred_lab = np.argmax(outputs, 1)
        return outputs, pred_lab

    def uncertainty(self, y):
        norm_sum = 0
        sum_arr = 0
        for i in range(len(y)):
            norm_sum += np.linalg.norm(y[i])
            sum_arr += y[i]
        sum_norm = np.linalg.norm(sum_arr)
        diff = (1 / len(y)) * (norm_sum - sum_norm)
        return diff

    def uncertainty_1(self, x):
        l2 = np.linalg.norm(x, axis=2)
        norm_sum = np.sum(l2, axis=1)
        x_sum = np.sum(x, axis=1)
        sum_norm = np.linalg.norm(x_sum, axis=1)
        diff = (1/len(x)) * (norm_sum-sum_norm)
        return diff

    @staticmethod
    def sel_target_class(target_class, train_labels, length):
        y = map(lambda x: x if x == target_class else 100, train_labels)
        y_new = np.array([item for item in y])
        idx = [i for i in range(length) if y_new[i] != 100]
        return idx

    def train_consistency_model(self, train_data, train_labels, specific_class):
        """
        For consistency model, concatenate the train and test data, for the MNIST dataset
        :param train_data:
        :param train_labels:
        :param specific_class:
        :return:
        """
        sample_num = 20000  # MNIST
        train_data_num = train_data.shape[0]

        if specific_class == 10:
            idx_seldom_class = []
            idx = [i for i in range(train_data_num)]
            np.random.shuffle(idx)
            selected_id = idx[0:sample_num]
            selected_id.extend(idx_seldom_class)
            x_all = train_data[selected_id]
            y_all = train_labels[selected_id]
            x_all = x_all.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        else:
            y = map(lambda x: x if x==specific_class else 10, train_labels)
            y_new = np.array([item for item in y])
            idx = [i for i in range(50000) if y_new[i] != 10]
            idx_remain = [i for i in range(50000) if i not in idx]
            np.random.shuffle(idx_remain)
            idx_remain_sel = idx_remain[0:8000]
            idx_new = idx+idx_remain_sel
            np.random.shuffle(idx_new)
            x_all = train_data[idx_new]
            y_all = y_new[idx_new]

        outputs = self.encoder_model.get_latent(x_all)
        print(outputs.shape)
        if not type(outputs) is np.ndarray:
            outputs = outputs.eval()

        consistency_model = LabelSpreading(gamma=6)
        consistency_model.fit(outputs, y_all)

        return consistency_model

    def train_consistency_model_balanced(self, train_data, train_labels, class1=None, class2=None, binary=False):
        """
        For consistency model, concatenate the train and test data
        :param train_data:
        :param train_labels:
        :param class1 :
        :param class2: the manifold is between class_1 and class_2
        :return:
        """
        sample_num = 30000  # cicids
        train_data_num = train_data.shape[0]
        seldom_class = [1, 8, 9, 11, 12]
        remain_class = list(set(list(range(13))) - set(seldom_class))
        idx_seldom_class = []
        sample_num_per_class = int(sample_num / len(remain_class))
        selected_id = []

        if binary:
            for target_class in [0, 1]:
                idx_target_class = self.sel_target_class(target_class, train_labels, train_data_num)
                np.random.shuffle(idx_target_class)
                selected_id.extend(idx_target_class[0: 10000])

            x_all = train_data[selected_id]
            y_all = train_labels[selected_id]
            x_all = x_all.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        else:
            if class1 is None:
                for target_class in seldom_class:
                    idx_seldom_class.extend(self.sel_target_class(target_class, train_labels, train_data_num))
                for target_class in remain_class:
                    idx_target_class = self.sel_target_class(target_class, train_labels, train_data_num)
                    np.random.shuffle(idx_target_class)
                    selected_id.extend(idx_target_class[0: sample_num_per_class])

                selected_id.extend(idx_seldom_class)
                x_all = train_data[selected_id]
                y_all = train_labels[selected_id]
                x_all = x_all.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            else:
                idx_class1 = self.sel_target_class(class1, train_labels, train_data_num)
                np.random.shuffle(idx_class1)
                selected_id.extend(idx_class1[0: sample_num_per_class])

                idx_class2 = self.sel_target_class(class2, train_labels, train_data_num)
                np.random.shuffle(idx_class2)
                selected_id.extend(idx_class2[0: sample_num_per_class])

                x_all = train_data[selected_id]
                y_all = train_labels[selected_id]

        outputs = self.encoder_model.get_latent(x_all)
        print(outputs.shape)
        if not type(outputs) is np.ndarray:
            outputs = outputs.eval()

        consistency_model = LabelSpreading(kernel='knn', n_neighbors=6, n_jobs=-1)
        # consistency_model = LabelSpreading(gamma=3)
        consistency_model.fit(outputs, y_all)

        return consistency_model

