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

if __name__ == '__main__':
    unittest.main()
