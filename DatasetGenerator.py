import numpy as np


def gp_datagen(n, seed=0):
    np.random.seed(seed)
    m = 4 * (np.random.random((2, 2)) - 0.5)
    s = 2 * np.random.random((2, 2))
    y = np.random.randint(0, 2, n)
    return np.array(
        list(zip(np.random.normal(m[y], s[y]), y))
    )

