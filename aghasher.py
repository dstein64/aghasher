import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io
import utils

utils.checknumpy() # not necessary since already called... (when import util)

class AnchorGraphHasher:
  def __init__(self, W, anchors, nnanchors, sigma):
    self.W = W
    self.anchors = anchors
    self.nnanchors = nnanchors
    self.sigma = sigma
    
  @classmethod
  def train(cls, traindata, anchors, numhashbits=12,
            nnanchors=2, sigma=None):
    m = anchors.shape[0]
    # numhashbits must be less than num anchors because we get m-1 eigenvalues from an m-by-m matrix. (m-1 since we omit eigenvalue=1)
    if numhashbits >= m:
      valerr = 'The number of hash bits (%s) must be less than the number of anchors (%s).' % (numhashbits, m)
      raise ValueError(valerr)
    (Z, sigma) = cls._Z(traindata, anchors, nnanchors, sigma)
    W = cls._W(Z, numhashbits)
    Y = cls._hash(Z, W)
    agh = cls(W, anchors, nnanchors, sigma)
    return (agh, Y)
  
  def hash(self, data):
    (Z, _) = self._Z(data, self.anchors, self.nnanchors, self.sigma)
    return self._hash(Z, self.W)
  
  @staticmethod
  def _hash(Z, W):
    Y = Z.dot(W)
    return Y > 0
  
  @staticmethod
  def test(trainY, testY, traingnd, testgnd, radius=2):
    # make sure trangnd and testgnd are flattened
    testgnd = testgnd.ravel()
    traingnd = traingnd.ravel()
    ntest = testY.shape[0]
    
    # Here's how to use Hamming dist built into numpy, but it is twice as slow as yours
    #dim = trainY.shape[1]
    #hamdis = scipy.spatial.distance.cdist(trainY, testY, 'hamming')    
    #hamdis = dim * hamdis
    #hamdis = (hamdis+.1).astype(np.int) # convert to int but make sure no rounding issues by adding .1 first
    
    hamdis = utils.pdist2(trainY, testY, 'hamming')
    
    precision = np.zeros(ntest)
    for j in xrange(ntest):
      ham = hamdis[:,j]
      lst = np.flatnonzero(ham <= radius)
      ln = len(lst)
      if ln == 0:
        precision[j] = 0
      else:
        precision[j] = len(np.flatnonzero(traingnd[lst] == testgnd[j])) / float(ln)
        
    return np.mean(precision)
  
  @staticmethod
  def _W(Z, numhashbits):
    s = np.asarray(Z.sum(0)).ravel() # extra steps here are for compatibility with sparse matrices
    isrl = np.diag(np.power(s, -0.5)) # isrl = inverse square root of lambda
    ztz = Z.T.dot(Z) # ztz = Z transpose Z
    if scipy.sparse.issparse(ztz):
      ztz = ztz.todense()
    M = np.dot(isrl, np.dot(ztz, isrl))
    eigenvalues, V = scipy.linalg.eig(M) # there is also a numpy.linalg.eig
    I = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[I]
    V = V[:,I]
    
    # this is also essentially what they do in the matlab since check for equality to 1
    # doesn't work because of floating point precision
    if eigenvalues[0] > 0.99999999:
      eigenvalues = eigenvalues[1:]
      V = V[:,1:]
    eigenvalues = eigenvalues[0:numhashbits]
    V = V[:,0:numhashbits]
    # paper also multiplies by sqrt(n), but their matlab code doesn't. isn't necessary.
    
    W = np.dot(isrl, np.dot(V, np.diag(np.power(eigenvalues, -0.5))))
    return W
  
  @staticmethod
  def _Z(data, anchors, nnanchors, sigma):
    n = data.shape[0]
    m = anchors.shape[0]
    
    # tried using for loops. too slow.
    #sqdist = scipy.spatial.distance.cdist(data, anchors,'sqeuclidean') # too slow
    sqdist = utils.pdist2(data, anchors, 'sqeuclidean')
    val = np.zeros((n, nnanchors))
    pos = np.zeros((n, nnanchors), dtype=np.int)
    for i in range(nnanchors):
      pos[:,i] = np.argmin(sqdist, 1)
      val[:,i] = sqdist[np.arange(len(sqdist)), pos[:,i]]
      sqdist[np.arange(n), pos[:,i]] = float('inf')
    
    # would be cleaner to calculate sigma in its own separate method, but this is more efficient
    if sigma is None:
      dist = np.sqrt(val[:,nnanchors-1])
      sigma = np.mean(dist) / np.sqrt(2)
    
    # Next, calculate formula (2) from the paper
    # this calculation differs from the matlab. In the matlab, the RBF kernel's exponent only
    # has sigma^2 in the denominator. Here, 2 * sigma^2. This is accounted for when auto-calculating sigma
    # above by dividing by sqrt(2)
    
    # Here is how you first calculated formula (2), which is similar in approach to the matlab code (not with
    # respect to the difference mentioned above, though). However, you encountered floating point issues.
    #val = np.exp(-val / (2 * np.power(sigma,2)))
    #s = val.sum(1)[np.newaxis].T # had to do np.newaxis and transpose to make it a column vector
    #                             # just calling ".T" without that wasn't working. reshape would
    #                             # also work. I'm not sure which is preferred.
    #repmat = np.tile(s, (1,nnanchors))
    #val = val / repmat

    # So work in log space and then exponentiate, to avoid the floating point issues.
    # for the denominator, the following code avoids even more precision issues, by relying
    # on the fact that the log of the sum of exponentials, equals some constant plus the log of sum
    # of exponentials of numbers subtracted by the constant:
    #  log(sum_i(exp(x_i))) = m + log(sum_i(exp(x_i-m)))
    
    c = 2 * np.power(sigma,2) # bandwidth parameter
    exponent = -val / c       # exponent of RBF kernel
    # no longer using the np.newaxis approach, since you now see keepdims option
    #shift = np.amin(exponent, 1)[np.newaxis].T # np.axis to make column vector
    shift = np.amin(exponent, 1, keepdims=True)
    # no longer using the np.tile approach, since numpy figures it out. You were originally doing
    # exponent - shiftrep, but exponent - shift works the same.
    #shiftrep = np.tile(shift, (1,nnanchors))
    denom = np.log(np.sum(np.exp(exponent - shift), 1, keepdims=True)) + shift
    val = np.exp(exponent - denom)
    
    Z = scipy.sparse.lil_matrix((n,m))
    for i in range(nnanchors):
      Z[np.arange(n), pos[:,i]] = val[:,i]
    Z = scipy.sparse.csr_matrix(Z)
    
    return (Z, sigma)

if __name__ == '__main__':
  
  datadir = utils.diroffile(__file__) + '/../data/'
  
  mnist_split = scipy.io.loadmat(datadir + 'mnist_split.mat')
  anchor_300  = scipy.io.loadmat(datadir + 'anchor_300.mat')
  
  traindata = mnist_split['traindata']
  testdata  = mnist_split['testdata']
  traingnd  = mnist_split['traingnd']
  testgnd   = mnist_split['testgnd']
  anchors   = anchor_300['anchor']
  
  precisionradius = 2 # hamming radius 2 precision
  sigma = None # sigma None means sigma auto-calculated
  nnanchors = 2 # number-of-nearest anchors
  
  for numbits in [12,16,24,32,48,64]:
    agh, trainY = AnchorGraphHasher.train(traindata, anchors, numbits, nnanchors, sigma)
    testY = agh.hash(testdata)
    precision = AnchorGraphHasher.test(trainY, testY, traingnd, testgnd, precisionradius)
    print '1-AGH: the Hamming radius %d precision for %d bits is %f.' % (precisionradius, numbits, precision)
  