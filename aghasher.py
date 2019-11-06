import os

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io

import utils


class AnchorGraphHasher:
    def __init__(self, W, anchors, nn_anchors, sigma):
        self.W = W
        self.anchors = anchors
        self.nn_anchors = nn_anchors
        self.sigma = sigma

    @classmethod
    def train(cls, X, anchors, num_hashbits=12, nn_anchors=2, sigma=None):
        m = anchors.shape[0]
        # num_hashbits must be less than num anchors because we get m-1
        # eigenvalues from an m-by-m matrix.
        # (m-1 since we omit eigenvalue=1)
        if num_hashbits >= m:
            valerr = (
                'The number of hash bits ({}) must be less than the number of '
                'anchors ({}).'
            ).format(num_hashbits, m)
            raise ValueError(valerr)
        Z, sigma = cls._Z(X, anchors, nn_anchors, sigma)
        W = cls._W(Z, num_hashbits)
        Y = cls._hash(Z, W)
        agh = cls(W, anchors, nn_anchors, sigma)
        return agh, Y

    def hash(self, X):
        Z, _ = self._Z(X, self.anchors, self.nn_anchors, self.sigma)
        return self._hash(Z, self.W)

    @staticmethod
    def _hash(Z, W):
        Y = Z.dot(W)
        return Y > 0

    @staticmethod
    def test(Y_train, Y_test, T_train, T_test, radius=2):
        # Flatten arrays
        T_test = T_test.ravel()
        T_train = T_train.ravel()
        ntest = Y_test.shape[0]

        hamdis = utils.pdist2(Y_train, Y_test, 'hamming')

        precision = np.zeros(ntest)
        for j in range(ntest):
            ham = hamdis[:, j]
            lst = np.flatnonzero(ham <= radius)
            ln = len(lst)
            if ln == 0:
                precision[j] = 0
            else:
                numerator = len(np.flatnonzero(T_train[lst] == T_test[j]))
                precision[j] = numerator / float(ln)

        return np.mean(precision)

    @staticmethod
    def _W(Z, num_hashbits):
        # The extra steps here are for compatibility with sparse matrices.
        s = np.asarray(Z.sum(0)).ravel()
        isrl = np.diag(np.power(s, -0.5))  # isrl = inverse square root of lambda
        ztz = Z.T.dot(Z)  # ztz = Z transpose Z
        if scipy.sparse.issparse(ztz):
            ztz = ztz.todense()
        M = np.dot(isrl, np.dot(ztz, isrl))
        eigenvalues, V = scipy.linalg.eig(M)  # there is also a numpy.linalg.eig
        I = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[I]
        V = V[:, I]

        # This is also essentially what they do in the matlab reference, since a check for
        # equality to 1 doesn't work because of floating point precision.
        if eigenvalues[0] > 0.99999999:
            eigenvalues = eigenvalues[1:]
            V = V[:, 1:]
        eigenvalues = eigenvalues[0:num_hashbits]
        V = V[:, 0:num_hashbits]
        # The paper also multiplies by sqrt(n), but their matlab reference code doesn't.
        # It isn't necessary.

        W = np.dot(isrl, np.dot(V, np.diag(np.power(eigenvalues, -0.5))))
        return W

    @staticmethod
    def _Z(X, anchors, nn_anchors, sigma):
        n = X.shape[0]
        m = anchors.shape[0]

        sqdist = utils.pdist2(X, anchors, 'sqeuclidean')
        val = np.zeros((n, nn_anchors))
        pos = np.zeros((n, nn_anchors), dtype=np.int)
        for i in range(nn_anchors):
            pos[:, i] = np.argmin(sqdist, 1)
            val[:, i] = sqdist[np.arange(len(sqdist)), pos[:, i]]
            sqdist[np.arange(n), pos[:, i]] = float('inf')

        if sigma is None:
            dist = np.sqrt(val[:, nn_anchors - 1])
            sigma = np.mean(dist) / np.sqrt(2)

        # Calculate formula (2) from the paper. This calculation differs from the reference matlab.
        # In the matlab, the RBF kernel's exponent only has sigma^2 in the denominator. Here, 2 * sigma^2.
        # This is accounted for when auto-calculating sigma above by dividing by sqrt(2).

        # Work in log space and then exponentiate, to avoid the floating point issues. For the
        # denominator, the following code avoids even more precision issues, by relying on the fact that
        # the log of the sum of exponentials, equals some constant plus the log of sum of exponentials
        # of numbers subtracted by the constant:
        #  log(sum_i(exp(x_i))) = m + log(sum_i(exp(x_i-m)))

        c = 2 * np.power(sigma, 2)  # bandwidth parameter
        exponent = -val / c  # exponent of RBF kernel
        shift = np.amin(exponent, 1, keepdims=True)
        denom = np.log(np.sum(np.exp(exponent - shift), 1, keepdims=True)) + shift
        val = np.exp(exponent - denom)

        Z = scipy.sparse.lil_matrix((n, m))
        for i in range(nn_anchors):
            Z[np.arange(n), pos[:, i]] = val[:, i]
        Z = scipy.sparse.csr_matrix(Z)

        return Z, sigma


if __name__ == '__main__':
    datadir = os.path.join(utils.diroffile(__file__))

    mnist_split = scipy.io.loadmat(os.path.join(datadir, 'mnist_split.mat'))
    anchor_300 = scipy.io.loadmat(os.path.join(datadir, 'anchor_300.mat'))

    X_train = mnist_split['traindata']
    X_test = mnist_split['testdata']
    T_train = mnist_split['traingnd']
    T_test = mnist_split['testgnd']
    anchors = anchor_300['anchor']

    radius = 2  # hamming radius 2 precision
    sigma = None  # sigma None means sigma auto-calculated
    nnanchors = 2  # number-of-nearest anchors

    for numbits in [12, 16, 24, 32, 48, 64]:
        agh, Y_train = AnchorGraphHasher.train(
            X_train, anchors, numbits, nnanchors, sigma)
        Y_test = agh.hash(X_test)
        precision = AnchorGraphHasher.test(
            Y_train, Y_test, T_train, T_test, radius)
        print(f'1-AGH: the Hamming radius {radius} precision for {numbits} bits is {precision}.')
