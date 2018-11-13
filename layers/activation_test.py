import unittest
import numpy as np
from layers.activation import *


class TestActivation(unittest.TestCase):
    def test_relu(self):
        layer = Activation('relu')
        x = np.array([-5.5, 2.4, -0.5, 3.6])
        y = layer.forward(x)
        delta_y = np.array([-1.2, 1.7, 0.5, -2.6])
        delta_x = layer.backward(delta_y)

        np.testing.assert_array_equal(delta_x, np.array([0, 1.7, 0, -2.6]))


if __name__ == '__main__':
    unittest.main()
