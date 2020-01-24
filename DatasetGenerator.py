import numpy as np


def gp_datagen(N):
    m = 10 * (np.random.random((2, 2)) - 0.5)
    s = 2 * np.random.random((2, 2))
    y = np.random.randint(0, 2, N)
    return np.array(
        list(zip(np.random.normal(m[y], s[y]), y))
    )

