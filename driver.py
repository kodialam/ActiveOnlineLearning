from DatasetGenerator import gp_datagen
from OnlineClassifiers import train_al_perceptron
from GraphingUtils import plot_points_and_separators
from ALRules import *

import matplotlib.pyplot as plt
import numpy as np
import random

N_train = 1000
N_test = 100


def acc(classifier):
    return classifier.score(
        np.stack(data_test[:, 0]),
        data_test[:, 1].astype(int)
    )


data = gp_datagen(N_train + N_test)
data_train = data[:N_train]
data_test = data[N_train:]

clf_all, ixs_all = train_al_perceptron(data_train)
clfs = {
    'all': clf_all,
}

for p in np.linspace(0, 1, 8):
    if 1 > p > 0:
        clf_random, ixs_random = train_al_perceptron(
            data_train,
            rule=lambda *x: random.random() < p
        )
        clfs['random {:0.2f}'.format(p)] = clf_random

for b in np.linspace(1, 20, 4):
    clf_bs, ixs_bs = train_al_perceptron(
        data_train,
        rule=b_sampling(b)
    )
    if len(ixs_bs) > 0:
        clfs['b-sampling {:0.2f}'.format(b)] = clf_bs

for g in np.linspace(1, 20, 4):
    clf_lm, ixs_lm = train_al_perceptron(
        data_train,
        rule=lm_sampling(g)
    )
    if len(ixs_lm) > 0:
        clfs['lm-sampling {:0.2f}'.format(g)] = clf_lm


plot_points_and_separators(data_test, clfs)