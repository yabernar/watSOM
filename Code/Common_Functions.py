import numpy as np


def quadratic_distance(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sum((x - y) ** 2)


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2) / sigma
    # Between 0 and 1/sig


def normalized_gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2)
    # Between 0 and 1


def euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))


def normalized_euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))/np.sqrt(x.shape[0])
