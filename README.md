aghasher
========

An implementation of the Anchor Graph Hashing algorithm (AGH-1), presented in *Hashing with Graphs* (Liu et al. 2011).

Dependencies
------------

*aghasher* requires Python 2.7 with numpy and scipy. These should be linked with a BLAS implementation (e.g., OpenBLAS,
ATLAS, Intel MKL). Without being linked to BLAS, numpy/scipy will use a fallback that causes PyAnchorGraphHasher to run
over 50x slower.

How To Use
----------

To use aghasher, first import the *aghasher* module.

    import aghasher
    
### Training a Model

An AnchorGraphHasher is constructed using the *train* method, which returns an AnchorGraphHasher and the hash bit
embedding for the training data.

    agh, y_train = aghasher.AnchorGraphHasher.train(train_data, anchors, num_bits, nn_anchors, sigma)

AnchorGraphHasher.train takes 5 arguments:

* **train_data** An *n-by-d* numpy.ndarray with training data. The rows correspond to observations, and the columns
  correspond to dimensions.
* **anchors** An *m-by-d* numpy.ndarray with anchors. *m* is the total number of anchors. Rows correspond to anchors,
  and columns correspond to dimensions. The dimensionality of the anchors much match the dimensionality of the training
  data.
* **num_bits** (optional; defaults to 12) Number of hash bits for the embedding.
* **nn_anchors** (optional; defaults to 2) Number of nearest anchors that are used for approximating the neighborhood
  structure.
* **sigma** (optional; defaults to *None*) sigma for the Gaussian radial basis function that is used to determine
  similarity between points. When sigma is specified as *None*, the code will automatically set a value, depending on
  the training data and anchors.

### Hashing Data with an AnchorGraphHasher Model

With an AnchorGraphHasher object, which has variable name *agh* in the preceding and following examples, hashing
out-of-sample data is done with the object's *hash* method.

    agh.hash(data)
    
The hash method takes one argument:

* **data** An *n-by-d* numpy.ndarray with data. The rows correspond to observations, and the columns correspond to
dimensions. The dimensionality of the data much match the dimensionality of the training data used to train the
AnchorGraphHasher.

Since Python does not have a native bit vector data structure, the hash method returns an *n-by-r* numpy.ndarray, where
*n* is the number of observations in *data*, and *r* is the number of hash bits specified when the model was trained.
The elements of the returned array are boolean values that correspond to bits.

### Testing an AnchorGraphHasher Model

Testing is performed with the AnchorGraphHasher.test method.

    precision = AnchorGraphHasher.test(y_train, y_test, t_train, t_test, radius)
    
AnchorGraphHasher.test takes 5 arguments:

* **y_train** An *n-by-r* numpy.ndarray with the hash bit embedding corresponding to the training data. The rows
  correspond to the *n* observations, and the columns correspond to the *r* hash bits.
* **y_test** An *m-by-r* numpy.ndarray with the hash bit embedding corresponding to the testing data. The rows
  correspond to the *m* observations, and the columns correspond to the *r* hash bits.
* **t_train** An *n-by-1* numpy.ndarray with the ground truth labels for the training data.
* **t_test** An *m-by-1* numpy.ndarray with the ground truth labels for the testing data.
* **radius** (optional; defaults to 2) Hamming radius to use for calculating precision.

### Executing aghasher.py

*aghasher.py* can be executed from the command line.

    $ python aghasher.py
    
Running aghasher.py runs the \__main__ code, which uses an AnchorGraphHasher to replicate the training/testing performed
in the Matlab code.

The code in the \__main__ section serves as an example of how to use AnchorGraphHasher.

Differences from the Matlab Reference Implementation
----------------------------------------------------

The code is structured differently than the Matlab reference implementation.

The Matlab code implements an additional hashing method, hierarchical hashing (referred to as 2-AGH), an extension of
1-AGH that is not implemented here.

There is one functional difference relative to the Matlab code. If *sigma* is specified (as opposed to being
auto-estimated), then for the same value of *sigma*, the Matlab and Python code will produce different results. They
will produce the same results when the Matlab *sigma* is sqrt(2) times bigger than the manually specified *sigma* in the
Python code. This is because in the Gaussian RBF kernel, the Python code uses a 2 in the denominator of the exponent,
and the Matlab code does not. A 2 was included in the denominator of the Python code, as that is the canonical way to
use an RBF kernel.

License
-------

*aghasher* has an [MIT License](https://en.wikipedia.org/wiki/MIT_License).

See [LICENSE](LICENSE).

References
----------

Liu, Wei, Jun Wang, Sanjiv Kumar, and Shih-Fu Chang. 2011. “Hashing with Graphs.” In Proceedings of the 28th
International Conference on Machine Learning (ICML-11), edited by Lise Getoor and Tobias Scheffer, 1–8. ICML ’11. New
York, NY, USA: ACM.
