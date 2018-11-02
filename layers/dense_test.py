import unittest
import numpy as np
from layers.dense import Dense
from ops.activations import *


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        layer = Dense(3, 2, 'sigmoid')

        # change random kernels to a fixed kernels
        layer.kernels = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        x = np.array([[0.1, 0.2, 0.3]])

        actual = layer.forward(x)
        expected = sigmoid(np.dot(x, layer.kernels))

        np.testing.assert_array_equal(expected, actual)


    def test_forward_without_activation(self):
        layer = Dense(3, 2, use_bias=False)

        # change random kernels to a fixed kernels
        layer.kernels = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        x = np.array([[0.1, 0.2, 0.3]])

        actual = layer.forward(x)
        expected = np.dot(x, layer.kernels)

        np.testing.assert_array_equal(expected, actual)

    def test_forward_with_bias(self):
        layer = Dense(3, 2, use_bias=True)
        pass

if __name__ == '__main__':
    unittest.main()
