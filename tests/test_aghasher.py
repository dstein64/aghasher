import os
import unittest

import numpy as np

from aghasher import AnchorGraphHasher


class TestAghasher(unittest.TestCase):
    def test_aghasher(self):
        with np.load(os.path.join(os.path.dirname(__file__), 'data.npz')) as data:
            X_train = data['X_train']
            X_test = data['X_test']
            T_train = data['T_train']
            T_test = data['T_test']
            anchors = data['anchors']

        radius = 2  # hamming radius 2 precision
        sigma = None  # sigma None means sigma auto-calculated
        nn_anchors = 2  # number-of-nearest anchors

        # Maps number of bits to expected precision.
        expected_precision_lookup = {
            12: 0.6990520452891523,
            16: 0.7785310451160524,
            24: 0.8551476847057504,
            32: 0.8735075625522515,
            48: 0.8769409352078942,
            64: 0.8767468477596928,
        }

        for num_bits, expected_precision in expected_precision_lookup.items():
            agh, Y_train = AnchorGraphHasher.train(
                X_train, anchors, num_bits, nn_anchors, sigma)
            Y_test = agh.hash(X_test)
            precision = AnchorGraphHasher.test(
                Y_train, Y_test, T_train, T_test, radius)
            self.assertAlmostEqual(precision, expected_precision)


if __name__ == '__main__':
    unittest.main()
