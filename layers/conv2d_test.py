import unittest
import numpy as np
from layers.conv2d import Conv2D


class Conv2DTest(unittest.TestCase):
    def test_forward_simple(self):
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

    def test_forward_stride_2_padding_valid(self):
        layer = Conv2D(input_shape=(5, 5, 1), num_kernel=1, kernel_size=(3, 3), stride=(2, 2), padding='valid')
        x = np.empty((1, 5, 5, 1))
        x[0, :, :, 0] = np.array([
            [0, 1, 2, 1, 1],
            [2, 1, 1, 2, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 0, 0, 2],
            [0, 2, 0, 1, 0]
        ])
        w = np.empty((1, 3, 3, 1))
        w[0, :, :, 0] = np.array([
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0]
        ])
        expected = np.empty((1, 2, 2, 1))
        expected[0, :, :, 0] = np.array([
            [5, 5],
            [2, 3]
        ])
        layer.build()
        layer.kernel = w
        actual = layer.forward(x)

        np.testing.assert_array_equal(expected, actual)

    def test_forward_stride_1_padding_same(self):
        layer = Conv2D(input_shape=(3, 3, 1), num_kernel=1, kernel_size=(3, 3), stride=(1, 1), padding='same')
        x = np.empty((1, 3, 3, 1))
        x[0, :, :, 0] = np.array([
            [0, 1, 2],
            [1, 1, 2],
            [2, 0, 1]
        ])
        w = np.empty((1, 3, 3, 1))
        w[0, :, :, 0] = np.array([
            [0, 1, 2],
            [1, 2, 1],
            [1, 1, 1]
        ])
        expected = np.empty((1, 3, 3, 1))
        expected[0, :, :, 0] = np.array([
            [2, 7, 9],
            [6, 12, 9],
            [6, 7, 5]
        ])
        layer.build()
        layer.kernel = w
        actual = layer.forward(x)

        np.testing.assert_array_equal(expected, actual)

    def test_forward_2_channel(self):
        layer = Conv2D(input_shape=(3, 3, 2), num_kernel=1, kernel_size=(2, 2), stride=(1, 1), padding='valid')
        layer.build()
        x = np.empty((1, 3, 3, 2))
        x[0, :, :, 0] = np.array([
            [1,1,1],
            [1,1,1],
            [1,1,1]
        ])
        x[0, :, :, 1] = np.array([
            [2,2,2],
            [2,2,2],
            [2,2,2]
        ])

        w = np.empty((1, 2, 2, 2))
        w[0, :, :, 0] = np.array([
            [1,1],
            [1,1]
        ])
        w[0, :, :, 1] = np.array([
            [1,1],
            [1,1]
        ])
        layer.kernel = w

        expected = np.empty((1, 2, 2, 1))
        expected[0, :, :, 0] = np.array([
            [12, 12],
            [12, 12]
        ])
        actual = layer.forward(x)

        np.testing.assert_array_equal(expected, actual)



if __name__ == '__main__':
    unittest.main()
