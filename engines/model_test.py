import unittest
import numpy as np

from layers.dense import Dense
from layers.conv2d import Conv2D
from engines.model import Model


class TestModel(unittest.TestCase):
    def test_can_not_fit_if_was_not_compiled(self):
        nn = Model([Dense(1,1)])
        with self.assertRaises(RuntimeError) as cm:
            nn.fit([1,2], 1)

        self.assertTrue(isinstance(cm.exception, RuntimeError))

    def test_must_fail_if_layers_is_not_a_list(self):
        with self.assertRaises(RuntimeError) as cm:
            Model("123")

        self.assertTrue(isinstance(cm.exception, RuntimeError))

    def test_must_fail_if_layers_contains_non_layer_objects(self):
        with self.assertRaises(RuntimeError) as cm:
            Model(["123"])

        self.assertTrue(isinstance(cm.exception, RuntimeError))

    def test_should_fail_if_add_non_layer_object(self):
        nn = Model()
        with self.assertRaises(RuntimeError) as cm:
            nn.add('something')

        self.assertTrue(isinstance(cm.exception, RuntimeError))

    def test_model_should_work(self):
        nn = Model([Dense(3, 1)])
        nn.compile('mean_squared_error')
        x = np.array([
            [1, 2, 3]
        ])
        y_one_hot = np.array([
            [1]
        ])
        nn.fit(x, y_one_hot, n_epochs=1, learning_rate=0.001)
        nn.predict(x)

    def test_model_with_conv2d_should_work(self):
        nn = Model([Conv2D(input_shape=(3, 3, 1), num_kernel=1, kernel_size=(3,3))])
        nn.compile(loss='mean_squared_error')
        x = np.zeros((1, 3, 3, 1))
        y = np.ones((1, 1, 1, 1))
        nn.fit(x, y, n_epochs=1)


if __name__ == '__main__':
    unittest.main()
