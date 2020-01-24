from sklearn.linear_model import SGDClassifier


def train_al_perceptron(data, rule=lambda *x: True):
    clf = SGDClassifier(
        loss="perceptron",
        eta0=1,
        learning_rate="constant",
        penalty=None
    )
    accepted = []
    for i, (x, y) in enumerate(data):
        if rule(x, clf):
            accepted.append(i)
            if len(accepted) == 1:
                clf.partial_fit([x], [y], classes=[0, 1])
            else:
                clf.partial_fit([x], [y])

    return clf, accepted


