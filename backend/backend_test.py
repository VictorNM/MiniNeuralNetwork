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

    def test_softmax(self):
        x = np.array([
            [10000, 1, 1],
            [1, 10000, 1]
        ])
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        actual = K.softmax(x)

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

    def test_categorical_crossentropy(self):
        y = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0]
        ])
        y_hat = np.array([
            [2413123, 1232, 2],
            [1321, 1012323, 123213],
            [3413231, 12312343, 231]
        ])
        actual = K.categorical_crossentropy(y, y_hat)
        d_actual = K.categorical_crossentropy_derivative(y, y_hat)
        print(actual)
        print(d_actual)


class ConvTest(unittest.TestCase):
    def test_simple_input_convolutions(self):
        x = np.empty((1, 3, 3, 1))
        x[0, :, :, 0] = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        actual = K.compute_input_convolutions(x, kernel_size=(2, 2), output_size=(2, 2))
        expected = np.empty((1, 4,4,1))
        expected[0, :, :, 0] = np.array([
            [5, 6, 8, 9],
            [4, 5, 7, 8],
            [2, 3, 5, 6],
            [1, 2, 4, 5]
        ])
        np.testing.assert_array_equal(expected, actual)

    def test_input_convolutions_input_3x4_kernel_2x2(self):
        x = np.empty((1, 3, 4, 1))
        x[0, :, :, 0] = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        actual = K.compute_input_convolutions(x, kernel_size=(2, 2), output_size=(2, 3), stride=(1, 1))
        expected = np.empty((1, 4, 6, 1))
        expected[0, :, :, 0] = np.array([
            [6, 7, 8, 10, 11, 12],
            [5, 6, 7, 9, 10, 11],
            [2, 3, 4, 6, 7, 8],
            [1, 2, 3, 5, 6, 7]
        ])
        np.testing.assert_array_equal(expected, actual)

    def test_input_convolutions_stride_2(self):
        x = np.empty((1, 4, 4, 1))
        x[0, :, :, 0] = np.array([
            [3, 1, 0, 1],
            [1, 1, 2, 0],
            [1, 2, 2, 1],
            [0, 1, 0, 2]
        ])
        actual = K.compute_input_convolutions(x, kernel_size=(3,3), output_size=(1, 1), stride=(2, 2))
        expected = np.empty((1, 9, 1, 1))
        expected[0, :, :, 0] = np.array([
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

    def test_kernel_convolutions(self):
        w = np.empty((1, 2, 2, 1))
        w[0, :, :, 0] = np.array([
            [1, 2],
            [3, 4]
        ])

        actual = K.compute_kernel_convolutions(w, input_size=(3, 3), output_size=(2, 2))
        expected = np.empty((1, 4, 9, 1))
        expected[0, :, :, 0] = np.array([
            [4, 3, 0, 2, 1, 0, 0, 0, 0],
            [0, 4, 3, 0, 2, 1, 0, 0, 0],
            [0, 0, 0, 4, 3, 0, 2, 1, 0],
            [0, 0, 0, 0, 4, 3, 0, 2, 1]
        ])
        np.testing.assert_array_equal(expected, actual)

    def test_kernel_convolutions_stride_2(self):
        w = np.empty((1, 2, 2, 1))
        w[0, :, :, 0] = np.array([
            [1, 2],
            [3, 4]
        ])
        actual = K.compute_kernel_convolutions(w, input_size=(3, 3), output_size=(1, 1), stride=(2, 2))
        expected = np.empty((1, 1, 9, 1))
        expected[0, :, :, 0] = np.array([
            [4, 3, 0, 2, 1, 0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(expected, actual)


    def test_compute_delta_kernels(self):
        x_conv = np.empty((4, 4, 1))
        x_conv[:, :, 0] = np.array([
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1]
        ])
        delta_y_flatten = np.empty((4, 1))
        delta_y_flatten[:, 0] = np.array([
            [1, 1, 1, 1]
        ])
        expected = np.empty((1, 2, 2, 1))
        expected[0, :, :, 0] = np.array([
            [3, 3],
            [2, 2]
        ])
        actual = K.compute_delta_kernels(delta_y_flatten, x_conv, kernel_size=(2,2))

        np.testing.assert_array_equal(expected, actual)

    def test_compute_delta_kernels_2_kernels(self):
        x_conv = np.empty((4, 4, 1))
        x_conv[:, :, 0] = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        delta_y_flatten = np.empty((4, 2))
        delta_y_flatten[:, 0] = np.array([
            [1, 0, 1, 1]
        ])
        delta_y_flatten[:, 1] = np.array([
            [1, 0, 1, 1]
        ])
        expected = np.empty((2, 2, 2, 1))
        expected[0, :, :, 0] = np.array([
            [2, 1],
            [2, 0]
        ])
        expected[1, :, :, 0] = np.array([
            [2, 1],
            [2, 0]
        ])
        actual = K.compute_delta_kernels(delta_y_flatten, x_conv, kernel_size=(2, 2))

        np.testing.assert_array_equal(expected, actual)

    def test_compute_delta_kernels_2_channels(self):
        x_conv = np.empty((4, 4, 2))
        x_conv[:, :, 0] = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        x_conv[:, :, 1] = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        delta_y_flatten = np.empty((4, 1))
        delta_y_flatten[:, 0] = np.array([
            [1, 0, 1, 1]
        ])
        expected = np.empty((1, 2, 2, 2))
        expected[0, :, :, 0] = np.array([
            [2, 1],
            [2, 0]
        ])
        expected[0, :, :, 1] = np.array([
            [2, 1],
            [2, 0]
        ])
        actual = K.compute_delta_kernels(delta_y_flatten, x_conv, kernel_size=(2, 2))

        np.testing.assert_array_equal(expected, actual)

    def test_compute_delta_input(self):
        delta_y_flatten = np.empty((4, 1))
        delta_y_flatten[:, 0] = np.array([
            [0, 2, 2, 1]
        ])
        kernel_convolutions = np.empty((1, 4, 9, 1))
        kernel_convolutions[0, :, :, 0] = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ])
        expected = np.empty((3, 3, 1))
        expected[:, :, 0] = np.array([
            [0, 2, 2],
            [2, 5, 3],
            [2, 3, 1]
        ])
        actual = K.compute_delta_input(delta_y_flatten, kernel_convolutions, input_shape=(3, 3, 1))

        np.testing.assert_array_equal(expected, actual)

    def test_compute_delta_input_2_kernels(self):
        delta_y_flatten = np.empty((4, 2))
        delta_y_flatten[:, 0] = np.array([
            [0, 2, 2, 1]
        ])
        delta_y_flatten[:, 1] = np.array([
            [0, 2, 2, 1]
        ])
        kernel_convolutions = np.empty((2, 4, 9, 1))
        kernel_convolutions[0, :, :, 0] = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ])
        kernel_convolutions[1, :, :, 0] = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ])
        expected = np.empty((3, 3, 1))
        expected[:, :, 0] = np.array([
            [0, 4, 4],
            [4, 10, 6],
            [4, 6, 2]
        ])
        actual = K.compute_delta_input(delta_y_flatten, kernel_convolutions, input_shape=(3, 3, 1))

        np.testing.assert_array_equal(expected, actual)

    def test_compute_delta_input_2_channel(self):
        delta_y_flatten = np.empty((4, 1))
        delta_y_flatten[:, 0] = np.array([
            [0, 2, 2, 1]
        ])
        kernel_convolutions = np.empty((1, 4, 9, 2))
        kernel_convolutions[0, :, :, 0] = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ])
        kernel_convolutions[0, :, :, 1] = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ])
        expected = np.empty((3, 3, 2))
        expected[:, :, 0] = np.array([
            [0, 2, 2],
            [2, 5, 3],
            [2, 3, 1]
        ])
        expected[:, :, 1] = np.array([
            [0, 2, 2],
            [2, 5, 3],
            [2, 3, 1]
        ])
        actual = K.compute_delta_input(delta_y_flatten, kernel_convolutions, input_shape=(3, 3, 2))

        np.testing.assert_array_equal(expected, actual)

if __name__ == '__main__':
    unittest.main()
