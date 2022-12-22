import numpy as np
import random
from sklearn.semi_supervised import LabelSpreading
from scipy.special import comb
from scipy import stats
import tensorflow as tf

from utils.setup_NSL import NSLModel

modified_fea = [6, 7, 9, 12, 13, 14, 15, 18, 19] + list(range(20, 27)) + list(range(29, 37))


class Detector:
    def __init__(self, train_data, train_labels, features_num, model, max_drop_num, drop_variants, modify_value, threshold):
        self.features_num = features_num
        self.max_modify_num = max_drop_num
        possible_drop_variants = self.choice_num(features_num, max_drop_num)
        self.modify_variants = min(drop_variants, possible_drop_variants)
        self.threshold = threshold
        self.modify_type = 'modify'

        self.uncertain_eval_type = 'clf'
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.consist_model = self.train_consistency_model(train_data, train_labels)
        self.sigma = modify_value

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

    def dropout_features(self, x, drop_ls, max_drop_num, drop_num=False):
        n = x.shape[0]
        x = x.reshape(n, -1)
        if not drop_num:
            drop_num = random.randrange(1, max_drop_num)
        drop_idx = random.sample(drop_ls, drop_num)
        selected = [xx for xx in range(self.features_num) if xx not in drop_idx]
        remain_features_num = len(selected)
        x_new = np.zeros((x.shape[0], remain_features_num))
        for i in range(x.shape[0]):
            x_new[i, :] = x[i, selected]
        return x_new, drop_num

    def modify_features(self, x, modify_ls, max_modify_num, modify_num=False):
        x = x.reshape(1, -1)
        x_new = np.zeros((x.shape[0], x.shape[1]))
        if not modify_num:
            modify_num = random.randrange(1, max_modify_num)
        modified_idx = random.sample(modify_ls, modify_num)
        un_modified_idx = [xx for xx in range(self.features_num) if xx not in modified_idx]
        x_new[0, modified_idx] = np.clip(x[0, modified_idx] + np.random.normal(0, self.sigma), 0, 1)
        x_new[0, un_modified_idx] = x[0, un_modified_idx]
        x_new = x_new.reshape((1, self.features_num))
        return x_new

    def detect(self, x_test, nn_predict):
        """
        detect the adversarial example by evaluating model output uncertainty
        """
        adv_ls, pred_labels = [], []
        adv = False
        manifold_inconsis_num = 0
        diff = 0
        x = x_test.reshape(1, -1)

        #  step-1
        pred_consist = self.consistency_model(x)  # the output of consistency model
        pred_label_consist = np.argmax(pred_consist)
        manifold_clf_diff = self.uncertainty([pred_consist, nn_predict])
        if not pred_label_consist == np.argmax(nn_predict):
            adv = True
            manifold_inconsis_num += 1
            final_label = pred_label_consist
            print('manifold inconsistent')
        else:  # step-2
            x_new = np.zeros((self.modify_variants, self.features_num))
            for i in range(self.modify_variants):
                x_new[i] = self.modify_features(x[0], modified_fea, self.max_modify_num)

            y_pred, pred_labels = self.clf_output(x_new)
            y_pred_ls = list(y_pred)
            pred_labels = list(pred_labels)
            print(' predicted labels for {} noisy data: {}'.format(self.modify_variants, y_pred))
            print('model prediction on true input: \n predicted prob: {}'.format(nn_predict[0]))

            pred_labels.append(np.argmax(nn_predict))
            y_pred_ls.append(np.ravel(nn_predict))

            mode = stats.mode(pred_labels)[0][0]
            if not mode == 1 and 1 in pred_labels:
                y_pred_ls_new = []
                for i, label in enumerate(pred_labels):
                    if not label == 1:
                        y_pred_ls_new.append(y_pred_ls[i])
                pred_labels.remove(1)
            else:
                y_pred_ls_new = y_pred_ls

            diff = self.uncertainty(y_pred_ls_new)
            final_label = stats.mode(pred_labels)[0][0]

            if diff > self.threshold:
                adv = True

            print('predict labels: {}'.format(pred_labels))
            print('uncertainty： {} \n'.format(diff))
        return adv, pred_labels, final_label, diff, manifold_inconsis_num, manifold_clf_diff

    def detect_para(self, x_test, nn_predict):
        """
        detect the adversarial example by evaluating model output uncertainty
        """
        x = x_test.reshape(-1, self.features_num)

        #  step-1 manifold consist with the clf output or not
        pred_consist = self.consist_model.predict_proba(x)  # the output of consistency model
        preds = np.concatenate((pred_consist, nn_predict), 0)
        preds = preds.reshape((2, len(pred_consist), -1))
        preds = np.swapaxes(preds, 0, 1)
        mani_clf_dif = self.uncertainty_1(preds)

        # step 2
        filter = list(map(lambda x: 1 if x in modified_fea else 0, [i for i in range(self.features_num)]))
        x_noisy = np.minimum(np.maximum(x + filter * np.random.normal(loc=0, scale=self.sigma, size=x.shape), 0), 1)
        for i in range(self.modify_variants - 1):
            x_new = np.minimum(np.maximum(x + filter * np.random.normal(loc=0, scale=self.sigma, size=x.shape), 0), 1)
            x_noisy = np.concatenate((x_noisy, x_new), 0)

        y_pred, _ = self.clf_output(x_noisy)
        y_pred = np.concatenate((y_pred, nn_predict), 0)
        y_pred = y_pred.reshape((self.modify_variants + 1, len(x), y_pred.shape[1]))
        y_pred = np.swapaxes(y_pred, 0, 1)

        diff = self.uncertainty_1(y_pred)

        return mani_clf_dif, diff

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

    def train_consistency_model(self, train_data, train_labels):
        consist_model = LabelSpreading(gamma=3)
        consist_model.fit(train_data, train_labels)
        return consist_model

    def consistency_model(self,test_data):
        y_pred = self.consist_model.predict_proba(test_data)
        print('manifold prediction: {}\n'.format(y_pred))
        return y_pred
