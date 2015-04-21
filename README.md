PyAnchorGraphHasher
=================

An implementation of the Anchor Graph Hashing algorithm (AGH-1), presented in *Hashing with Graphs* (Liu et al. 2011).

Dependencies
------------

PyAnchorGraphHasher requires Python 2.7 with numpy and scipy. These should be linked with a BLAS implementation (e.g., OpenBLAS, ATLAS, Intel MKL). Without being linked to BLAS, numpy/scipy will use a fallback that causes PyAnchorGraphHasher to run over 50x slower. If BLAS is not linked, a warning message will be displayed when running the code.

The latter packages can be installed with pip.

    $ pip install numpy scipy

Or on Ubuntu using apt-get.

    $ apt-get install python-numpy python-scipy

Or on OS X using macports.

    $ port install py-numpy py-scipy

or brew.

    $ brew install numpy scipy

How To Use
----------

To use PyAnchorGraphHasher, first import the *aghasher* module.

    import aghasher
    
### Training a Model

An AnchorGraphHasher is constructed using the *train* method, which returns an AnchorGraphHasher and the hash bit embedding for the training data.

    (agh, trainY) = aghasher.AnchorGraphHasher.train(traindata, anchors, numbits, nnanchors, sigma)

AnchorGraphHasher.train takes 5 arguments:

* **traindata** An *n-by-d* numpy.ndarray with training data. The rows correspond to observations, and the columns correspond to dimensions.
* **anchors** An *m-by-d* numpy.ndarray with anchors. *m* is the total number of anchors. Rows correspond to anchors, and columns correspond to dimensions. The dimensionality of the anchors much match the dimensionality of the training data.
* **numbits** (optional; defaults to 12) Number of hash bits for the embedding.
* **nnanchors** (optional; defaults to 2) Number of nearest anchors that are used for approximating the neighborhood structure.
* **sigma** (optional; defaults to *None*) sigma for the Gaussian radial basis function that is used to determine similarity between points. When sigma is specified as *None*, the code will automatically set a value, depending on the training data and anchors.

### Hashing Data with an AnchorGraphHasher Model

With an AnchorGraphHasher object, which has variable name *agh* in the preceding and following examples, hashing out-of-sample data is done with the object's *hash* method.

    agh.hash(data)
    
The hash method takes one argument:

* **data** An *n-by-d* numpy.ndarray with data. The rows correspond to observations, and the columns correspond to dimensions. The dimensionality of the data much match the dimensionality of the training data used to train the AnchorGraphHasher.

Since Python does not have a native bit vector data structure, the hash method returns an *n-by-r* numpy.ndarray, where *n* is the number of observations in *data*, and *r* is the number of hash bits specified when the model was trained. The elements of the returned array are boolean values that correspond to bits.

### Testing an AnchorGraphHasher Model

Testing is performed with the AnchorGraphHasher.test method.

    precision = AnchorGraphHasher.test(trainY, testY, traingnd, testgnd, precisionradius)
    
AnchorGraphHasher.test takes 5 arguments:

* **trainY** An *n-by-r* numpy.ndarray with the hash bit embedding corresponding to the training data. The rows correspond to the *n* observations, and the columns correspond to the *r* hash bits.
* **testY** An *m-by-r* numpy.ndarray with the hash bit embedding corresponding to the testing data. The rows correspond to the *m* observations, and the columns correspond to the *r* hash bits.
* **traingnd** An *n-by-1* numpy.ndarray with the ground truth labels for the training data.
* **testgnd** An *m-by-1* numpy.ndarray with the ground truth labels for the testing data.
* **radius** (optional; defaults to 2) Hamming radius to use for calculating precision.

### Executing aghasher.py

*aghasher.py* can be executed from the command line.

    $ python aghasher.py
    
Running aghasher.py runs the \__main__ code, which uses an AnchorGraphHasher to replicate the training/testing performed in the Matlab code.

The code in the \__main__ section serves as an example of how to use AnchorGraphHasher.

Differences from the Matlab Implementation
------------------------------------------

The code is structured differently than the Matlab reference implementation.

The Matlab code implements an additional hashing method, hierarchical hashing, which is referred to as 2-AGH. 2-AGH is an extension of 1-AGH, and is currently not implemented in Python.

There is one functional difference relative to the Matlab code. If *sigma* is specified (as opposed to being auto-estimated), then for the same value of *sigma*, the Matlab and Python code will produce different results. They will produce the same results when the Matlab *sigma* is sqrt(2) times bigger than the manually specified *sigma* in the Python code. This is because in the Gaussian RBF kernel, the Python code uses a 2 in the denominator of the exponent, and the Matlab code does not. A 2 was included in the denominator of the Python code, as that is the canonical way to use an RBF kernel.

References
==========

Liu, Wei, Jun Wang, Sanjiv Kumar, and Shih-Fu Chang. 2011. “Hashing with Graphs.” In Proceedings of the 28th International Conference on Machine Learning (ICML-11), edited by Lise Getoor and Tobias Scheffer, 1–8. ICML ’11. New York, NY, USA: ACM.

