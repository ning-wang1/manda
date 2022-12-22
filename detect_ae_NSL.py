import os
import argparse
import numpy as np
import tensorflow as tf

# user define library
from utils.setup_NSL import NSL_KDD
from utils.setup_NSL import NSLModel
from utils import classifier as clf
from experiment_builder import AEDetect
from utils.classifier import evaluate_sub, get_success_advs, random_select, compute_roc_1, compute_roc
from utils.util_artifact import get_value, train_lr


def train_discriminators(train_data, train_labels):

    consist_model = clf.classifier('Consistency', train_data, train_labels)
    lgr_model = clf.classifier('LGR', train_data, train_labels)
    knn_model = clf.classifier('KNN', train_data, train_labels)
    bnb_model = clf.classifier('BNB', train_data, train_labels)
    svm_model = clf.classifier('SVM', train_data, train_labels)
    dtc_model = clf.classifier('DTC', train_data, train_labels)
    mlp_model = clf.classifier('MLP', train_data, train_labels)

    models = {'LGR': lgr_model, 'KNN': knn_model, 'BNB': bnb_model, 'SVM': svm_model, 'DTC': dtc_model,
              'MLP': mlp_model, 'consistency': consist_model}
    return models


def present_result(test_data, test_labels, model_dict):
    # the number of adversarial examples that models misclassified
    performance_dict = {}
    sample_num = test_labels.shape[0]
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        if model_name is 'Consistency':
            clf.evaluate_sub(model_name, test_labels, model.predict(test_data))
            pred = model.predict(test_data)
        elif model_name is 'NN':
            model.evaluate_only(test_data, test_labels)
            pred = model.predict(test_data).eval()
            pred = np.argmax(pred, axis=1)
        else:
            clf.evaluate_only(model_name, model, test_data, test_labels)
            pred = model.predict(test_data)
        pred =pred.reshape([-1, 1])
        correct_num = np.sum(pred==test_labels)
        misclassified_num = test_labels.shape[0] - correct_num
        performance_dict[model_name] = [correct_num, misclassified_num, sample_num]

    return performance_dict


def subset_data(data):
    # to deal with memory error (use small subset of data)
    print('Train data: {}; Test Data: {}'.format(data.train_data.shape, data.test_data.shape))
    train_data = data.train_data[0:10000, :]
    test_data = data.test_data[0:5000, :]
    train_labels_one_hot = data.train_labels[0:10000]
    test_labels_one_hot = data.test_labels[0:5000]
    train_labels = np.argmax(train_labels_one_hot, 1)
    test_labels = np.argmax(test_labels_one_hot, 1)
    return train_data, test_data, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot


def load_aes(attack_name):
    if 'cw' in attack_name:
        advs = np.load('data/crafted_ae/cw_nsl_5/Adv_nsl-kdd_cw-nsl_1.npy')
        true_labels = np.load('data/crafted_ae/cw_nsl_5/labels_nsl-kdd_cw-nsl_1.npy')
        for i in range(2,10):
            x = np.load('data/crafted_ae/cw_nsl_5/Adv_nsl-kdd_cw-nsl_{}.npy'.format(i+1))
            y = np.load('data/crafted_ae/cw_nsl_5/labels_nsl-kdd_cw-nsl_{}.npy'.format(i+1))
            print('success ae in 100 attempts', len(x))
            advs = np.concatenate((advs, x))
            true_labels = np.concatenate((true_labels, y))
    elif 'bim' in attack_name:
        advs = np.load('data/crafted_ae/Adv_nsl-kdd_bim-a-nsl.npy')
        true_labels = np.load('data/crafted_ae/labels_nsl-kdd_bim-a-nsl.npy')
    else:
        advs = np.load('data/crafted_ae/Adv_nsl-kdd_fgsm-nsl.npy')
        true_labels = np.load('data/crafted_ae/labels_nsl-kdd_fgsm-nsl.npy')

    return advs, true_labels


def detect_detail(args, samples, labs, labs_ae_flag):
    """
    detect AEs with the manifold inconsistency model
    :param samples:
    :param labs:
    :param labs_ae_flag:
    :return:
    """
    # reject AEs inferred by detector, eval acc on the detect as clean data
    adv_flag_ls, detect_as_clean_x, y, manifold_scores = AED.run_detect_detail(samples, labs)
    if not len(detect_as_clean_x) == 0:
        acc = model_to_attack.evaluate(detect_as_clean_x, y)
        print("Model accuracy on the purified (remove detect adv) test set: %0.2f%%" % (100 * acc))

    # eval detection performance
    evaluate_sub('Detection', labs_ae_flag, adv_flag_ls)
    fpr, tpr, auc_score = compute_roc_1(labs_ae_flag, manifold_scores, plot=False)
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    # Print and save results
    # ae_flags = labs_ae_flag.transpose()[0]
    # manifold = np.array(AED.manifold)
    # file_name = args.save_filepath + '_' + str(args.dataset) + '_' + str(args.attack) + '.txt'

    # with open(file_name, 'w') as f:
    #     print('the success rate of adversarial attack {}/{} : '.format(len(success_advs), len(adv_samples)), f)
    #     print('AE detected by manifold inconsistency: {} '.format(np.sum(ae_flags * manifold)), f)
    #
    #     for args.model_name in models.keys():
    #         misclassified_num = performance[args.model_name][1]
    #         print('# data  misclassified by the {} model:  {}/{} '.format(args.model_name, misclassified_num,
    #                                                                       len(samples)), f)
    return manifold_scores


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_filepath", type=str, default='result/uncertainty7')
    parser.add_argument("--model_name", type=str, default='nsl_kdd_Dos.h5')
    parser.add_argument("--modify_variants", type=int, default=3)
    parser.add_argument("--max_modify_num", type=int, default=15)
    parser.add_argument("--modify_value", type=int, default=0.005)
    parser.add_argument("--dataset", type=str, default='nsl-kdd')
    parser.add_argument("--attack", type=str, default='fgsm')
    parser.add_argument("--iter_n", type=int, default=1)
    args = parser.parse_args()

    return args


if __name__=='__main__':

    args = get_args()
    classifiers = {'KNN', 'LGR', 'BNB', 'DTC'}  # ML models
    attack_list = [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]  # attack classes

    # experiment setting (from possible selection)
    attack_class = [('DoS', 0.0)]
    # attack_class = [('Probe', 2.0)]
    # attack_class = [('R2L', 3.0)]
    # attack_class = [('U2R', 4.0)]

    # data pre-processing and data partition
    data = NSL_KDD(attack_class)
    train_data, test_data, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot = subset_data(data)
    features_num = train_data.shape[1]
    total_test_num = data.test_data.shape[0]

    # train models including SVM LG BNB DT
    model_dir = os.path.join("models", args.model_name)
    models = train_discriminators(train_data, train_labels)

    # load AEs
    adv_samples, labels = load_aes(args.attack)

    with tf.compat.v1.Session() as sess:

        # load model (victim model)
        model_to_attack = NSLModel(model_dir, features_num, sess)
        # models['NN'] = model_to_attack

        # get AEs that fools the model
        success_advs, labs_adv, pos = get_success_advs(model_to_attack.model, adv_samples, labels)

        # initial detector
        AED = AEDetect(args, data, features_num, train_data, train_labels, test_data, test_labels, test_labels_one_hot,
                       model_to_attack.model)

        # get clean data
        idxs = random_select(total_test_num, len(success_advs))
        clean_samples = data.test_data[idxs]
        labs_clean = data.test_labels[idxs]

        # concatenate AEs with clean data
        samples = np.concatenate((success_advs, clean_samples))
        labs = np.concatenate((labs_adv, labs_clean))  # true label
        labs_ae_flag = np.concatenate((np.ones([len(success_advs), 1]), np.zeros([len(clean_samples), 1])))  # ae:1

        # performance on other models (test AE transferbility)
        labels_numerical = np.argmax(labs, 1).reshape((labs.shape[0], -1))
        performance = present_result(samples[len(success_advs):], labels_numerical[len(success_advs):], models)

        # detect detail
        manifold_scores = detect_detail(args, samples[:10], labs[:10], labs_ae_flag[:10])
        # detect wit 2 step
        scores1, scores_2 = AED.run_detect_for_mixture(samples)
        # np.save('data/detect_result/1_scores_{}_{}.npy'.format(args.dataset, args.attack), scores1)
        # np.save('data/detect_result/2_scores_{}_{}.npy'.format(args.dataset, args.attack), scores_2)
