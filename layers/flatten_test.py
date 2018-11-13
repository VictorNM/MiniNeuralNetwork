import unittest
import numpy as np

from layers.flatten import Flatten


class FlattenTest(unittest.TestCase):
    def test_forward(self):
        layer = Flatten()
        x = np.zeros((2, 3, 4, 5))
        x[1, :, :, :] = np.ones((3, 4, 5))

        expected = x.reshape((2, 60))
        actual = layer.forward(x)

        np.testing.assert_array_equal(expected, actual)

    def test_forward_and_backward(self):
        layer = Flatten()
        x = np.zeros((2, 3, 4, 5))
        x[1, :, :, :] = np.ones((3, 4, 5))

        y = layer.forward(x)
        actual = layer.backward(y, 0.01)

        np.testing.assert_array_equal(x, actual)


if __name__ == '__main__':
    unittest.main()
