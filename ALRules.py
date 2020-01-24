import random
import numpy as np


def b_sampling(b):
    def df(x, clf):
        if hasattr(clf, "classes_"):
            dist = abs(clf.decision_function([x]))[0]
            return random.random() < b/(b + dist)
        return True
    return df


def lm_sampling(g):
    def df(x, clf):
        if hasattr(clf, "classes_"):
            dist = abs(clf.decision_function([x]))[0]
            return random.random() < np.exp(-g * dist)
        return True
    return df
