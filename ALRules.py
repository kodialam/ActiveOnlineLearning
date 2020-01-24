import random
import numpy as np


def predefined_func_decision(b, func):
    def df(x, clf):
        if hasattr(clf, "classes_"):
            dist = abs(clf.decision_function([x]))[0]
            return random.random() < func(b, dist)
        return True
    return df


def b_sampling(b, dist):
    return b / (b + dist)


def lm_sampling(b, dist):
    return np.exp(-b * dist)

