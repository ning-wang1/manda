import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import random
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from utils.classifier import compute_roc_1, compute_roc
from sklearn.metrics import precision_recall_curve
from utils.util_artifact import get_value, train_lr, normalize
from sklearn.preprocessing import scale
from scipy.interpolate import interp1d


def plot_roc(dataset, attack, type='roc'):
    pp = PdfPages('fig_{}_{}_{}.pdf'.format(type,dataset,attack))
    sns.set_style('darkgrid')
    plt.figure(figsize=(4.6, 3.4))

    th = 0.05
    th1 = 0.95

    # ------------------ our model --------------------------
    scores_1 = np.load('detect_result/1_scores_{}_{}.npy'.format(dataset, attack))
    scores_2 = np.load('detect_result/2_scores_{}_{}.npy'.format(dataset, attack))

    ae_flags = np.concatenate([np.ones(int(len(scores_1)/2)), np.zeros(int(len(scores_1)/2))])
    a = np.where(np.isnan(scores_1))
    b = np.where(np.isnan(scores_2))
    nan_idx = list(set(a[0]) | set(b[0]))
    scores_1 = np.delete(scores_1, nan_idx)
    scores_2 = np.delete(scores_2, nan_idx)
    ae_flags = np.delete(ae_flags, nan_idx)

    scores_1 = scale(scores_1)
    scores_2 = scale(scores_2)
    scores_2 = scores_2

    # ----------------------------lg model builder----------------------------------
    num_advs = int(len(scores_1)/2)
    idx_pos = np.where(ae_flags == 1)[0]
    idx_neg = np.where(ae_flags == 0)[0]
    num_sel = int(round(num_advs * 0.6))

    # Build detector
    _, _, lr = train_lr(
        densities_pos=scores_1[idx_pos[:num_sel]], densities_neg=scores_1[idx_neg[:num_sel]],
        uncerts_pos=scores_2[idx_pos[:num_sel]], uncerts_neg=scores_2[idx_neg[:num_sel]])

    # Evaluate detector
    values, labels = get_value(
        densities_pos=scores_1[idx_pos[num_sel:]], densities_neg=scores_1[idx_neg[num_sel:]],
        uncerts_pos=scores_2[idx_pos[num_sel:]], uncerts_neg=scores_2[idx_neg[num_sel:]])

    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]

    # Compute AUC
    n_samples = num_advs - num_sel

    # The first n_samples of 'probs' is  negative class (normal and noisy samples),
    fpr, tpr, auc_score = compute_roc(probs_pos=probs[n_samples:], probs_neg=probs[:n_samples], plot=False)
    print('MANDA Detector ROC-AUC score: %0.4f' % auc_score)
    plt.plot(fpr, tpr, ':', color='red', linewidth=2, label='$MANDA$ (Ours, AUC=%0.4f)' % auc_score)
    pos = np.where(fpr <= th)[0]
    print('TPR: {}'.format(tpr[pos[-1]]))

    pos1 = np.where(tpr >= th1)[0]
    print('FPR: {}'.format(fpr[pos1[0]]))

    # figure format
    if type == 'roc':
        plt.xlabel("FPR")
        plt.ylabel("TPR")
    else:
        plt.xlabel("Precision")
        plt.ylabel("Recall")
    plt.xlim((-0.02, 0.62))
    plt.legend(loc='lower right')
    if 'bim'in attack:
        plt.title('BIM attack')
    elif 'cw' in attack:
        plt.title('CW attack')
    elif 'fgsm' in attack:
        plt.title('FGSM attack')

    plt.grid(linestyle='--', linewidth=1)
    plt.tight_layout()
    pp.savefig()
    plt.show()
    plt.close()
    pp.close()


if __name__ == '__main__':
    # plot_roc('nsl-kdd', 'fgsm')
    # plot_roc('nsl-kdd', 'bim')
    # plot_roc('nsl-kdd', 'cw')
    # plot_roc('mnist', 'cw')
    # plot_roc('mnist', 'fgsm')
    # plot_roc('mnist', 'bim-a')
    # plot_roc('mnist', 'jsma')
    # plot_roc('CAN_DoS', 'jsma')
    plot_roc('cicids', 'fgsm-cicids')
    # plot_roc('cicids', 'bim-a-cicids')
    # plot_roc('cicids', 'cw-cicids')


