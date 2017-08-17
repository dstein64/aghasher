import numpy as np
import os

# scipy has a cdist function that works like matlab's pdist2 function.
# but for square euclidean distance it is slow for the version of scipy you have
# so implement pdist2.
# for details on its slowness, see https://github.com/scipy/scipy/issues/3251
# In your tests, it took over 16 seconds versus less than 4 seconds for the
# implementation below (where X has 69,000 elements and Y had 300)
# this has squared Euclidean distances.
def pdist2(X, Y, metric):
  metric = metric.lower()
  if metric == 'sqeuclidean':
    X = X.astype('float64')
    Y = Y.astype('float64')
    nx = X.shape[0]
    ny = Y.shape[0]
    XX = np.tile((X**2).sum(1),(ny,1)).T
    YY = np.tile((Y**2).sum(1),(nx,1))
    XY = X.dot(Y.T)
    sqeuc = XX + YY - 2*XY
    # Make negatives equal to zero. This arises due to floating point
    # precision issues. Negatives will be very close to zero (IIRC around
    # -1e-10 or maybe even closer to zero). Any better fix? you exhibited the
    #  floating point issue on two machines using the same code and data,
    # but not on a third. the inconsistent occurrence of the issue could
    # possibly be due to differences in numpy/blas versions across machines.
    return np.clip(sqeuc, 0, np.inf)
  elif metric == 'hamming':
    # scipy cdist supports hamming distance, but is twice as slow as yours
    # (even before multiplying by dim, and casting as int), possibly because
    # it supports non-booleans, but I'm not sure...
    # looping over data points in X and Y, and calculating hamming distance
    # to put in a hamdis matrix is too slow. this vectorized solution works
    # faster. separately, the matlab solution that uses compactbit is
    # approximately 8x more memory efficient, since 8 bits are required here
    # for each 0 or 1.
    hashbits = X.shape[1]
    Xint = (2 * X.astype('int8')) - 1
    Yint = (2 * Y.astype('int8')) - 1
    hamdis = hashbits - ((hashbits + Xint.dot(Yint.T)) / 2)
    return hamdis
  else:
    valerr = 'Unsupported Metric: %s' % (metric,)
    raise ValueError(valerr)

def pdist(X, metric):
  return pdist2(X, X, metric)

# Assumes columns contain variables/features, and rows contain
# observations/instances
def standardize(X):
  means = np.mean(X, 0, keepdims = True)
  stds = np.std(X, 0, keepdims = True)
  return (X - means) / stds

# return count of unique hashcodes in a hashcode matrix
def uniquehashcodes(X):
  s = set()
  for item in X[0:]:
    # any benefits/consequences to using .view instead of .astype ?
    s.add(tuple(np.packbits(item.astype(np.int8))))
  return len(s)

# return the directory of a file
def diroffile(f):
  return os.path.dirname(os.path.realpath(f))
