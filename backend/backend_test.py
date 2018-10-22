import unittest
import backend as K
import numpy as np


class ActivationTest(unittest.TestCase):
    def test_sigmoid_of_ten(self):
        expected = 0.9999546021
        actual = K.sigmoid(10)

        np.testing.assert_almost_equal(expected, actual)

    def test_sigmoid_of_array(self):
        expected = np.array([
            0.6224593312,
            0.7310585786,
            0.3775406688
        ])
        actual = K.sigmoid(np.array([0.5, 1, -0.5]))

        np.testing.assert_array_almost_equal(expected, actual)

    def test_d_sigmoid_of_two(self):
        expected = -2
        actual = K.d_sigmoid(2)

        np.testing.assert_almost_equal(expected, actual)


class LossTest(unittest.TestCase):
    def test_mse_scalar(self):
        expected = 2
        actual = K.mean_square_error(3, 5)

        np.testing.assert_equal(expected, actual)

    def test_mse_1d_array(self):
        y = np.array([1, 2, 3])
        y_hat = np.array([1, 2, 3])

        expected = 0
        actual = K.mean_square_error(y, y_hat)

        np.testing.assert_equal(expected, actual)

    def test_mse_2d_array(self):
        y = np.array([
            [1, 2, 3],
            [3, 2, 1]
        ])
        y_hat = np.array([
            [1, 2, 3],
            [3, 2, 1]
        ])

        expected = 0
        actual = K.mean_square_error(y, y_hat)

        np.testing.assert_equal(expected, actual)


class ConvTest(unittest.TestCase):
    def test_im2col_stride_1_padding_none(self):
        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        actual = K.im2col(x, (2, 2), stride=1, padding=None)
        expected = np.array([
            [5, 6, 8, 9],
            [4, 5, 7, 8],
            [2, 3, 5, 6],
            [1, 2, 4, 5]
        ])
        np.testing.assert_array_equal(expected, actual)

    def test_im2col_image_3x4_kernel_2x2(self):
        x = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        actual = K.im2col(x, (2, 2), stride=1, padding='valid')
        expected = np.array([
            [6, 7, 8, 10, 11, 12],
            [5, 6, 7, 9, 10, 11],
            [2, 3, 4, 6, 7, 8],
            [1, 2, 3, 5, 6, 7]
        ])
        np.testing.assert_array_equal(expected, actual)

    def test_should_fail_if_stride_differrent_to_1_in_mode_same_padding(self):
        x = np.array([[1,1],[1,1]])
        with self.assertRaises(RuntimeError) as cm:
            K.im2col(x, (2,2), stride=2, padding='same')

        self.assertTrue(isinstance(cm.exception, RuntimeError))

    def test_im2col_paddind_same(self):
        x = np.array([
            [3, 1, 0],
            [1, 1, 2],
            [1, 2, 2]
        ])
        w_shape = (3, 3)
        actual = K.im2col(x, w_shape, padding='same')
        expected = np.array([
            [1, 2, 0, 2, 2, 0, 0, 0, 0],
            [1, 1, 2, 1, 2, 2, 0, 0, 0],
            [0, 1, 1, 0, 1, 2, 0, 0, 0],
            [1, 0, 0, 1, 2, 0, 2, 2, 0],
            [3, 1, 0, 1, 1, 2, 1, 2, 2],
            [0, 3, 1, 0, 1, 1, 0, 1, 2],
            [0, 0, 0, 1, 0, 0, 1, 2, 0],
            [0, 0, 0, 3, 1, 0, 1, 1, 2],
            [0, 0, 0, 0, 3, 1, 0, 1, 1]
        ])

        np.testing.assert_array_equal(expected, actual)

    def test_im2col_stride_2(self):
        x = np.array([
            [3, 1, 0, 1],
            [1, 1, 2, 0],
            [1, 2, 2, 1],
            [0, 1, 0, 2]
        ])
        w_shape = (3, 3)
        actual = K.im2col(x, w_shape, stride=2, padding='valid')
        expected = np.array([
            [2],
            [2],
            [1],
            [2],
            [1],
            [1],
            [0],
            [1],
            [3]
        ])

        np.testing.assert_array_equal(expected, actual)

if __name__ == '__main__':
    unittest.main()
