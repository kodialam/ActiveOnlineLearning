from DatasetGenerator import gp_datagen
from OnlineClassifiers import train_al_perceptron
from GraphingUtils import *
from ALRules import *

import matplotlib.pyplot as plt
import numpy as np
import random

N_train = 1000
N_test = 100

data = gp_datagen(N_train + N_test, seed=2)
data_train = data[:N_train]
data_test = data[N_train:]


def acc(classifier):
    return classifier.score(
        np.stack(data_test[:, 0]),
        data_test[:, 1].astype(int)
    )


clfs = {}
n_pts = {}

for p in np.linspace(0, 1, 20):
    if p > 0:
        clf_random, ixs_random = train_al_perceptron(
            data_train,
            rule=lambda *x: random.random() < p
        )
        clfs['random {:0.2f}'.format(p)] = clf_random
        n_pts['random {:0.2f}'.format(p)] = len(ixs_random)

for b in np.logspace(0.001, 1000, 10):
    clf_bs, ixs_bs = train_al_perceptron(
        data_train,
        rule=predefined_func_decision(b, b_sampling)
    )
    if len(ixs_bs) > 0:
        clfs['b-sampling {:0.2f}'.format(b)] = clf_bs
        n_pts['b-sampling {:0.2f}'.format(b)] = len(ixs_bs)

for g in np.logspace(0.001, 1000, 10):
    clf_lm, ixs_lm = train_al_perceptron(
        data_train,
        rule=predefined_func_decision(g, lm_sampling)
    )
    if len(ixs_lm) > 0:
        clfs['lm-sampling {:0.2f}'.format(g)] = clf_lm
        n_pts['lm-sampling {:0.2f}'.format(g)] = len(ixs_lm)

accs = {k: acc(v) for k, v in clfs.items()}

plt.figure()
rule_types = (
    'random',
    'b-sampling',
    'lm-sampling'
)
for rule in rule_types:
    plt.plot(
        [v for k, v in n_pts.items() if rule in k],
        [v for k, v in accs.items() if rule in k],
        '-o', alpha=0.5
    )
plt.xlabel('Points Sampled')
plt.ylabel('Accuracy')
plt.legend(rule_types)
plt.show()

plot_points(data_test)
