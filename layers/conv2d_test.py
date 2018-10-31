import unittest
import numpy as np
from layers.conv2d import Conv2D


class Conv2DTest(unittest.TestCase):
    def test_forward(self):
        layer = Conv2D(
            input_shape=(4, 4, 1),
            num_kernel=1,
            kernel_size=(3,3)
        )
        x = np.empty((1, 4, 4, 1))
        x[0,:,:,0] = np.array([
            [3, 1, 0, 1],
            [1, 1, 2, 0],
            [1, 2, 2, 1],
            [0, 1, 0, 2]
        ])
        w = np.empty((1, 3, 3, 1))
        w[0, :, :, 0] = np.array([
            [1, 0, 2],
            [1, 2, 0],
            [0, 1, 1]
        ])
        layer.build()
        layer.kernel = np.array(w)
        actual = layer.forward(x)
        expected = np.empty((1, 2, 2, 1))
        expected[0, :, :, 0] = np.array([
            [12, 10],
            [8, 12]
        ])
        np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
