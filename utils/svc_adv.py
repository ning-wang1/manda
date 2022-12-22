import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
import scipy
from sklearn import datasets, linear_model, preprocessing
# from sklearn.datasets import fetch_mldata
import os.path
import random
import scipy
import scipy.optimize as opt
import re


def Dis(x, x1):
    d = x-x1
    D = ((d**2).sum())**0.5
    return D


def c_e(x1, y_prime, svc_model):
    y1 = [0, 0]
    y1[y_prime] = 1
    yh = svc_model.predict_proba(x1.reshape(1, -1))
    ce = -(y1*np.log(yh)).sum()
    return ce


def l_fun(x1, *args):
    x = args[0]
    y_prime = args[1]
    c = args[2]
    svc_model = args[3]
    l_f = c*Dis(x, x1)+c_e(x1, y_prime, svc_model)
    # l_f = c_e(x1,y_prime,svc_model)
    # print('loss ce:', c_e(x1,y_prime,svc_model))
    # print('loss dis:',Dis(x,x1))
    return l_f


def L_BFGS_B(x, y_prime, c, svc_model):
    initial = np.ones(x.shape)
    x1 = opt.fmin_l_bfgs_b(l_fun, x0=initial.flatten(), args=(x, y_prime, c, svc_model), approx_grad=True)
    x2 = x1[0]
    D = Dis(x, x2)
    yh = svc_model.predict(x2.reshape(1, -1))
    return x2, yh, D


def svm_adv(target, original, svc_model, x_test, y_test, c=0.1, E=0.1):
    # Find random instance of m in test set
    idx = np.random.randint(0, y_test.shape[0]-1)
    while y_test[idx] != original or svc_model.predict(x_test[idx, :].reshape(1, -1)) != original:
        if idx < y_test.shape[0]-1:
            idx += 1
        else:
            idx = np.random.randint(0, y_test.shape[0]-1)

    x = x_test[idx, :].reshape(1, -1)
    print('x_shape-----------.', x.shape)
    print('Benign svm_model prediction: ', svc_model.predict_proba(x))
 
    y_prime = target
    print('The target of adversary is y_prime: {}'.format(y_prime))
    # initialize C, get the largest c that can generate an adversarial example
    while c < 100:
        c = 2*c
        x_prime, yh, D = L_BFGS_B(x, y_prime, c, svc_model)
        print('yh------------', yh)
        if yh != y_prime:
            break
            
    # Bisection Search
    print('Bisection Search start!')
    c_low = 0 
    c_high = c
    while True:
        c_half = (c_high+c_low)/2
        x_prime, yh, D_prime = L_BFGS_B(x, y_prime, c_half, svc_model)
        print('C={}'.format(c_half))
        if yh != y_prime:
            D = D_prime
            c_high = c_half
        else:
            c_low = c_half
        if c_high - c_low < E:
            if yh != y_prime:
                x_prime, yh, D_prime = L_BFGS_B(x, y_prime, c/2, svc_model)
            else:
                break
    print(svc_model.predict(x_prime.reshape(1, -1)))
    return x_prime.reshape(1, -1)
