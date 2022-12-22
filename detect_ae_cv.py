import argparse
import tensorflow as tf

from utils.setup_mnist import MNISTModel, MNIST, MNISTAEModel
from utils.setup_cifar import CIFARModel, CIFAR, CIFARAEModel
from utils.setup_can import CAN, CANAEModel, CANModel
from utils.setup_cicids import CICIDS, CICIDSModel, CICIDSAEModel
import numpy as np
import math
import random
# from setup_inception import ImageNet, InceptionModel
from experiment_builder_cv import AEDetect
from sklearn import metrics
from utils.classifier import evaluate_sub, get_success_advs, random_select, compute_roc, compute_roc_1
from utils.classifier import get_correctly_pred_data


def detect_detail(samples, labs, labs_ae_flag):
    # reject AEs inferred by detector, eval acc on the detect as clean data
    detect_as_clean_x, y, manifold_scores = AED.run_detect_detail(samples, labs)
    if not len(detect_as_clean_x) == 0:
        acc = model_to_attack.evaluate(detect_as_clean_x, y)
        print("Model accuracy on the purified (remove detect adv) test set: %0.2f%%" % (100 * acc))

    # eval detection performance
    evaluate_sub('Detection', labs_ae_flag, AED.adv_flags[-len(labs_ae_flag):])
    fpr, tpr, auc_score = compute_roc_1(labs_ae_flag, manifold_scores, plot=True)
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    # Print and save results
    file_name = args.save_filepath.split('.')
    file_name = file_name[0] + '.txt'
    ae_flags = labs_ae_flag.transpose()[0]
    manifold = np.array(AED.manifold)[-len(labs_ae_flag):]

    with open(file_name, 'w') as f:
        print('the success rate of adversarial attack {}/{} : '.format(len(success_advs), len(adv_samples)), f)
        print("Only take a small sample from all the test records to show the detect detail")
        print('AE detected by manifold inconsistency for the sample: {}/{} '.format(np.sum(ae_flags * manifold),
                                                                                    len(samples)), f)

        print('Clean data that result in False Positive: {}/{} '.format(np.sum((1 - ae_flags) * manifold),
                                                                        len(samples)), f)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--save_filepath", type=str, default='result/uncertainty6.csv')
    parser.add_argument("--threshold", type=int, default=0.01)
    parser.add_argument("--modify_variants", type=int, default=3)
    parser.add_argument("--max_modify_num", type=int, default=200)
    parser.add_argument("--modify_value", type=int, default=0.015)  # mnist 0.3 # cicids 0.0006
    parser.add_argument("--attack", type=str, default='fgsm-cicids')
    parser.add_argument("--dataset", type=str, default='cicids')
    parser.add_argument('--change_threshold', type=float, default=0.01, help=' percentage of change in each feature')
    parser.add_argument('--intrusion_category', type=int, default=4)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    np.random.seed(1)
    tf.random.set_seed(1)
    args = get_args()
    args.dataset = 'cicids_binary'
    args.intrusion_category = 10

    if 'mnist' in args.dataset:
        data, model_dir, ae_model_dir = MNIST(), "models/mnist.h5", "models/mnist_ae.h5"
    elif 'cifar' in args.dataset:
        data, model_dir, ae_model_dir = CIFAR(), "models/cifar.h5", "models/cifar_ae.h5"
    elif 'CAN' in args.dataset:
        data, model_dir, ae_model_dir = CAN('DoS'), "models/can_DoS.h5", "models/can_DoS_ae.h5"
    elif args.dataset is 'cicids':
        data, model_dir = CICIDS(), "models/cicids.h5"
        ae_model_dir = "models/CICIDSAE.h5"
    elif 'cicids_binary' in args.dataset:
        data, model_dir = CICIDS(attack_cat=args.intrusion_category), 'models/cicids_binary.h5'
        ae_model_dir = "models/CICIDSAE.h5"

    # load generated AEs
    if 'cw' in args.attack:
        file_name = '{}_{}_{}_combined.npy'.format(args.dataset, args.attack, args.change_threshold)
    else:
        file_name = '{}_{}_{}_1.npy'.format(args.dataset, args.attack, args.change_threshold)
    adv_samples = np.load('data/Adv_' + file_name)
    labels = np.load('data/labels_' + file_name)

    with tf.compat.v1.Session() as sess:
        # load model
        if 'mnist' in args.dataset:
            model_to_attack = MNISTModel(model_dir, sess)
            cae = MNISTAEModel(ae_model_dir, sess)
        elif 'cifar' in args.dataset:
            model_to_attack = CIFARModel(model_dir, sess)
            cae = CIFARAEModel(ae_model_dir, sess)
        elif 'CAN' in args.dataset:
            model_to_attack = CANModel(model_dir, sess)
            cae = CANAEModel(ae_model_dir, sess)
        elif 'cicids' in args.dataset:
            model_to_attack = CICIDSModel(model_dir, sess)
            cae = CICIDSAEModel(ae_model_dir, sess)
        elif 'cicids_binary' in args.dataset:
            model = CICIDSModel(model_dir, sess, binary=True)
            cae = CICIDSAEModel(ae_model_dir, sess)

        # initial a detector
        AED = AEDetect(args, data, model_to_attack.model, cae)

        # get AEs that fools the model
        success_advs, labs_adv, _ = get_success_advs(model_to_attack.model, adv_samples, labels, target=0)

        # get clean data
        X_test, Y_test, x_remain, y_remain, _ = get_correctly_pred_data(model_to_attack.model,
                                                                        data.test_data, data.test_labels)
        total_num = X_test.shape[0]
        idxs = random_select(total_num, len(success_advs))
        clean_samples = X_test[idxs]
        labs_clean = Y_test[idxs]

        # concatenate AEs with clean data
        samples = np.concatenate((success_advs, clean_samples))
        labs = np.concatenate((labs_adv, labs_clean))  # true label
        labs_ae_flag = np.concatenate((np.ones([len(success_advs), 1]), np.zeros([len(clean_samples), 1])))  # ae:1

        # detect_detail is to show the detail of detection scheme
        # detect_detail(samples[400:450], labs[400:450], labs_ae_flag[400:450])
        # detect_detail(samples[-50:], labs[-50:], labs_ae_flag[-50:])
        # detect_detail(samples, labs, labs_ae_flag)

        # getting the scores for a mixture of clean data and adversarial data
        n_per_iter = 1000
        iters = math.ceil((len(samples)/n_per_iter))
        if iters > 1:
            scores_1, scores_2 = AED.run_detect_for_mixture(samples[:n_per_iter], dataset=args.dataset)
            i = 0
            for i in range(1, iters-1):
                print('detecting {}/{}'.format(n_per_iter*i, len(samples)))
                s_1, s_2 = AED.run_detect_for_mixture(samples[i*n_per_iter: (i+1)*n_per_iter], dataset=args.dataset)
                scores_1 = np.concatenate((scores_1, s_1))
                scores_2 = np.concatenate((scores_2, s_2))
            s_1, s_2 = AED.run_detect_for_mixture(samples[(i + 1) * n_per_iter:], dataset=args.dataset)
            scores_1 = np.concatenate((scores_1, s_1))
            scores_2 = np.concatenate((scores_2, s_2))
        else:
            scores_1, scores_2 = AED.run_detect_for_mixture(samples, dataset=args.dataset)

        # np.save('data/detect_result/1_scores_{}_{}.npy'.format(args.dataset, args.attack), scores_1)
        # np.save('data/detect_result/2_scores_{}_{}.npy'.format(args.dataset, args.attack), scores_2)

        np.save('data/singleclassadvscicids/1_scores_{}_{}.npy'.format(args.dataset, args.attack), scores_1)
        np.save('data/singleclassadvscicids/2_scores_{}_{}.npy'.format(args.dataset, args.attack), scores_2)
