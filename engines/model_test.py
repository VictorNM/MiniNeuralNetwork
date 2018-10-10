import unittest
import numpy as np

from layers.dense import Dense
from engines.model import Model
from ops.activations import *


class MyTestCase(unittest.TestCase):
    def test_backward_one_layer(self):
        y = np.array([
            [1, 0]
        ])

        y_hat = np.array([
            [0.98, 0.1]
        ])

        x = np.array([
            [0.5, 0.2, 0.9]
        ])

        w = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])

        layer = Dense(3, 2, activation='sigmoid', use_bias=False)
        layer.cache = x
        layer.weight = w
        layer.outputs = y_hat
        layers = [
            layer
        ]

        model = Model(layers)
        actual = [adjustment['delta_weight'] for adjustment in model.do_backward(y)]
        expected = np.array([[
            [-1.960e-04, 4.500e-03],
            [-7.840e-05, 1.800e-03],
            [-3.528e-04, 8.100e-03]
        ]])
        np.testing.assert_almost_equal(expected, actual)

    def test_backward_two_layer(self):
        # layer 1
        layer_1 = Dense(3, 2, activation='sigmoid', use_bias=False)
        layer_1.weight = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        layer_1.cache = np.array([
            [0.5, 0.2, 0.9]
        ])
        layer_1.outputs = np.array([
            [0.4, 0.9]
        ])

        # layer 2
        layer_2 = Dense(2, 2, activation='sigmoid', use_bias=False)
        layer_2.weight = np.array([
            [2, 5],
            [3, 4]
        ])
        layer_2.cache = np.array([
            [0.4, 0.9]
        ])
        layer_2.outputs = y_hat = np.array([
            [0.98, 0.1]
        ])

        # define network
        layers = [layer_1, layer_2]
        model = Model(layers)

        y = np.array([
            [1, 0]
        ])

        actual = [adjustment['delta_weight'] for adjustment in model.do_backward(y)]
        expected = [
            np.array([
                [0.00530592, 0.00156708],
                [0.00212237, 0.00062683],
                [0.00955066, 0.00282074]
            ]),
            np.array([
                [-0.0001568, 0.0036],
                [-0.0003528, 0.0081]
            ])
        ]
        for i in range(len(actual)):
            np.testing.assert_almost_equal(expected[i], actual[i])

    def test_backward_three_layers(self):
        # define layer 1
        layer_1 = Dense(3, 3, activation='sigmoid', use_bias=False)
        layer_1.weight = np.array([
            [1, 2, 2],
            [3, 4, 2],
            [5, 6, 2]
        ])
        layer_1.cache = np.array([
            [0.5, 0.2, 0.9]
        ])
        layer_1.outputs = np.array([
            [0.4, 0.9, 1.0]
        ])

        # define layer 2
        layer_2 = Dense(3, 2, activation='sigmoid', use_bias=False)
        layer_2.weight = np.array([
            [2, 5],
            [3, 4],
            [1, 4]
        ])
        layer_2.cache = np.array([
            [0.4, 0.9, 1.0]
        ])
        layer_2.outputs = np.array([
            [0.98, 0.1]
        ])

        # define layer 3
        layer_3 = Dense(2, 3, activation='sigmoid', use_bias=False)
        layer_3.weight = np.array([
            [0.5, 2.0, 1.0],
            [1.0, 4.1, 2.8]
        ])
        layer_3.cache = np.array([
            [0.98, 0.1]
        ])
        layer_3.outputs = np.array([
            [1.1, 2.2, 3.3]
        ])

        # define y
        y = np.array([
            [1.1, 2.4, 2.9]
        ])


        # define network
        layers = [layer_1, layer_2, layer_3]
        model = Model(layers)
        actual = [adjustment['delta_weight'] for adjustment in model.do_backward(y)]
        expected = [
            np.array([
                [-0.35145792, -0.10788228, 0.],
                [-0.14058317, -0.04315291, 0.],
                [-0.63262426, -0.1941881,  0.]
            ]),
            np.array([
                [-0.0155232, -0.228096],
                [-0.0349272, -0.513216],
                [-0.038808, -0.57024]
            ]),
            np.array([
                [0., 0.51744, -2.97528],
                [0., 0.0528, -0.3036]
            ])
        ]
        for i in range(len(actual)):
            np.testing.assert_almost_equal(expected[i], actual[i])

if __name__ == '__main__':
    unittest.main()
