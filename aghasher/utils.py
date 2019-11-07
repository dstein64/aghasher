import os

import numpy as np


def pdist2(X, Y, metric):
    # scipy has a cdist function that works like matlab's pdist2 function.
    # For square euclidean distance it is slow for the version of scipy you have.
    # For details on its slowness, see https://github.com/scipy/scipy/issues/3251
    # In your tests, it took over 16 seconds versus less than 4 seconds for the
    # implementation below (where X has 69,000 elements and Y had 300).
    # (this has squared Euclidean distances).
    metric = metric.lower()
    if metric == 'sqeuclidean':
        X = X.astype('float64')
        Y = Y.astype('float64')
        nx = X.shape[0]
        ny = Y.shape[0]
        XX = np.tile((X ** 2).sum(1), (ny, 1)).T
        YY = np.tile((Y ** 2).sum(1), (nx, 1))
        XY = X.dot(Y.T)
        sqeuc = XX + YY - 2 * XY
        # Make negatives equal to zero. This arises due to floating point
        # precision issues. Negatives will be very close to zero (IIRC around
        # -1e-10 or maybe even closer to zero). Any better fix? you exhibited the
        # floating point issue on two machines using the same code and data,
        # but not on a third. the inconsistent occurrence of the issue could
        # possibly be due to differences in numpy/blas versions across machines.
        return np.clip(sqeuc, 0, np.inf)
    elif metric == 'hamming':
        # scipy cdist supports hamming distance, but is twice as slow as yours
        # (even before multiplying by dim, and casting as int), possibly because
        # it supports non-booleans, but I'm not sure...
        # Looping over data points in X and Y, and calculating hamming distance
        # to put in a hamdis matrix is too slow. This vectorized solution works
        # faster.
        hashbits = X.shape[1]
        # Use high bitwidth int to prevent overflow (i.e., as opposed to int8
        # which could result in overflow when hashbits >= 64).
        X_int = (2 * X.astype('int')) - 1
        Y_int = (2 * Y.astype('int')) - 1
        hamdis = hashbits - ((hashbits + X_int.dot(Y_int.T)) / 2)
        return hamdis
    else:
        valerr = 'Unsupported Metric: %s' % (metric,)
        raise ValueError(valerr)


def standardize(X):
    # Assumes columns contain variables/features, and rows contain
    # observations/instances.
    means = np.mean(X, 0, keepdims=True)
    stds = np.std(X, 0, keepdims=True)
    return (X - means) / stds
