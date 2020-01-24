import matplotlib.pyplot as plt
import numpy as np


def plot_points_and_separators(data_test, clfs):
    plt.scatter(
        np.stack(data_test[:, 0])[:, 0],
        np.stack(data_test[:, 0])[:, 1],
        c=data_test[:, 1].astype(int)
    )
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    contours = {}
    contour_handles = []
    for c in clfs:
        contours[c] = clfs[c].decision_function(xy).reshape(XX.shape)
        contour = ax.contour(
            XX, YY,
            contours[c],
            levels=[0], alpha=0.5,
            linestyles=['--'],
            colors=[np.random.rand(3,)]
        )
        contour_handles.append(contour.legend_elements()[0][0])
    ax.legend(contour_handles, list(clfs.keys()))
    plt.show()

def plot_sampling_prob(decision_func):
    param_range = np.linspace(0, 10, 10)
    x = plt.xrange(-20, 20, 100)
    plt.figure()
    for param in param_range:
        def df_loc(x):
            return decision_func()
        plt.plot(x, np.vectorize(dec))
