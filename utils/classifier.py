from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
import pandas as pd
import random

import seaborn as sns
import warnings


def classifier(classifier_name, X_train, Y_train):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if classifier_name == 'KNN':
        # Train KNeighborsClassifier Model
        KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
        KNN_Classifier.fit(X_train, Y_train)
        model = KNN_Classifier
    elif classifier_name == 'LGR':
        # Train LogisticRegression Model
        LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
        LGR_Classifier.fit(X_train, Y_train)
        model = LGR_Classifier
    elif classifier_name == 'BNB':
        # Train Gaussian Naive Bayes Model
        BNB_Classifier = BernoulliNB()
        BNB_Classifier.fit(X_train, Y_train)
        model = BNB_Classifier
    elif classifier_name == 'DTC':
        # Train Decision Tree Model
        DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
        DTC_Classifier.fit(X_train, Y_train)
        model = DTC_Classifier
    elif classifier_name == 'SVM':
        SVC_Classifier = SVC(probability=True,  kernel="rbf")
        SVC_Classifier.fit(X_train, Y_train)
        model = SVC_Classifier
    elif classifier_name == 'MLP':
        MLP_Classifier = MLP(hidden_layer_sizes=(50,))
        MLP_Classifier.fit(X_train, Y_train)
        model = MLP_Classifier
    elif classifier_name == 'Consistency':
        consist_model = LabelSpreading(kernel='rbf', gamma=3)
        consist_model.fit(X_train, Y_train)
        model = consist_model
    else:
        print('ERROR: Unrecognized type of classifier')
    # evaluate(classifier_name, model, X_train, Y_train)
    return model


def evaluate(classifier_name, model, X, Y):
    scores = cross_val_score(model, X, Y, cv=5)
    Y_pre = model.predict(X)
    evaluate_sub(classifier_name, Y, Y_pre)
    print("Cross Validation Mean Score:" "\n", scores.mean())


def evaluate_only(classifier_name, model, X, Y):
    Y_pre = model.predict(X)
    evaluate_sub(classifier_name, Y, Y_pre)


def evaluate_sub(classifier_name, Y, Y_pre):
    accuracy = metrics.accuracy_score(Y, Y_pre)
    confusion_matrix = metrics.confusion_matrix(Y, Y_pre)
    classification = metrics.classification_report(Y, Y_pre)
    print()
    print('============================== {} Model Evaluation =============================='.format(classifier_name))
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()


def get_success_advs(model, samples, true_labels, target=None):
    preds = model(samples).eval()
    if target is None:
        pos1 = np.where(np.argmax(preds, 1) != np.argmax(true_labels, 1))
    else:
        pos1 = np.where(np.argmax(preds, 1) == target)
    x_sel = samples[pos1]
    y_sel = true_labels[pos1]

    return x_sel, y_sel, pos1


def get_correctly_pred_data(model, x, y, target=None, target_not=None):
    # sample idx that is correctly predicted
    preds = model(x).eval()
    pos1 = np.where(np.argmax(preds, 1) == np.argmax(y, 1))
    x_sel_correct = x[pos1]
    y_sel_correct = y[pos1]

    if target is not None:
        # sample idx that is correctly predicted and matched to the target categories
        pos11 = np.where(np.argmax(y_sel_correct, 1) == target)
        x_sel = x_sel_correct[pos11]
        y_sel = y_sel_correct[pos11]
    elif target_not is not None:
        # sample idx that is correctly predicted and matched to the target categories
        pos11 = np.where(np.argmax(y_sel_correct, 1) != target_not)
        x_sel = x_sel_correct[pos11]
        y_sel = y_sel_correct[pos11]
    else:
        x_sel = x_sel_correct
        y_sel = y_sel_correct

    # the samples that are not correctly predicted
    pos2 = np.where(preds.argmax(1) != y.argmax(1))
    x_remain = x[pos2]
    y_remain = y[pos2]
    return x_sel, y_sel, x_remain, y_remain, pos1


def random_select(max, num):
    lst = [i for i in range(max)]
    random.shuffle(lst)
    idxs = lst[0:num]
    return idxs


def plot_setting(self):
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    # np.set_printoptions(threshold=np.nan)
    np.set_printoptions(precision=3)
    sns.set(style="darkgrid")
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


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
        plt.style.use('seaborn-dark')
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        # plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def compute_roc_1(y, scores, plot=False):
    """
    TODO
    :param y:
    :param scores:
    :param plot:
    :return:
    """

    fpr, tpr, thresholds = roc_curve(y, scores)
    auc_score = auc(fpr, tpr)
    if plot:
        pp = PdfPages('test1.pdf')
        # plt.style.use('seaborn-dark')
        plt.figure(figsize=(6, 4.5))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        # plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid(linestyle='--', linewidth=1)
        # plt.show()
        pp.savefig()
        plt.close()
        pp.close()
    return fpr, tpr, auc_score


