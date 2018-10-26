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
        x = np.array([
            [3, 1, 0, 1],
            [1, 1, 2, 0],
            [1, 2, 2, 1],
            [0, 1, 0, 2]
        ])
        w = np.array([
            [1, 0, 2],
            [1, 2, 0],
            [0, 1, 1]
        ]).reshape(3,3,1)
        layer.build()
        layer.kernel = np.array([w])
        actual = layer.forward(x.reshape((1,4,4,1)))
        expected = np.array([
            [12, 10],
            [8, 12]
        ]).reshape(1,2,2,1)
        np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
