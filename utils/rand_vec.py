import numpy as np


def rand_vec(dim, cov):
    r = np.random.multivariate_normal([0]*dim, cov)
    r = np.matrix.transpose(np.matrix(r))
    return r


def rand_arr(length, cov):
    if length == 0:
        return []
    else:
        return np.random.multivariate_normal([0.]*length, cov)
