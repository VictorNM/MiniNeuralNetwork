import unittest
import numpy as np
from layers.dense import Dense
from ops.activations import *


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        layer = Dense(2, 'sigmoid', input_shape=(3,))

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
        layer = Dense(2, use_bias=False, input_shape=(3,))

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

    def test_should_build_successfully(self):
        layer = Dense(units=2)
        layer.build((1, 3))

        kernels_shape = np.shape(layer.kernels)

        self.assertTupleEqual(kernels_shape, (3, 2))

    def test_should_build_successfully_for_3d_input(self):
        layer = Dense(units=2)
        layer.build((1, 3, 2))

        kernels_shape = np.shape(layer.kernels)

        self.assertTupleEqual(kernels_shape, (6, 2))

if __name__ == '__main__':
    unittest.main()
