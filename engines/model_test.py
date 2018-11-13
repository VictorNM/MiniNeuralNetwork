import unittest
import numpy as np

from layers.dense import Dense
from layers.conv2d import Conv2D
from layers.flatten import Flatten
from engines.model import Model
from utils import train_utils

from keras.datasets import mnist
from sklearn.metrics import accuracy_score


class TestModel(unittest.TestCase):
    def test_can_not_fit_if_was_not_compiled(self):
        nn = Model([Dense(units=1)])
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
        nn = Model([Dense(units=1)])
        nn.compile('categorical_crossentropy')
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

    def test_combine_layers_should_work(self):
        nn = Model([
            Conv2D(input_shape=(6, 4, 1), num_kernel=3, kernel_size=(3, 3), use_bias=False, kernel_initializer='zeros'),
            Flatten(),
            Dense(units=2, use_bias=False)
        ])
        nn.compile(loss='categorical_crossentropy')
        x0 = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ]).reshape((6, 4, 1))
        x1 = np.array([
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]).reshape((6, 4, 1))
        x = np.array([x0, x1])
        y = [0, 1]
        nn.fit(x, train_utils.to_categorical(y, 2), n_epochs=100, learning_rate=0.001)

        x0_test = np.array([
            [1, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ]).reshape((6, 4, 1))
        x_test = np.array([x0_test])
        pred = nn.predict(x_test)
        print(pred)

    def test_with_small_mnist(self):
        num_classes = 10
        epochs = 12

        # input image dimensions
        img_rows, img_cols, num_channels = 28, 28, 1
        input_shape = (img_rows, img_cols, num_channels)

        # load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # reduce samples
        num_train_samples = 100
        x_train = x_train[:num_train_samples]
        y_train = y_train[:num_train_samples]

        num_test_samples = int(0.3 * num_train_samples)
        x_test = x_test[:num_test_samples]
        y_test = y_test[:num_test_samples]

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # normalize
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = train_utils.to_categorical(y_train, num_classes)

        model = Model()
        model.add(Conv2D(num_kernel=5, kernel_size=(3, 3), activation='relu', use_bias=False, kernel_initializer='zeros'))
        model.add(Flatten())
        model.add(Dense(units=num_classes, kernel_initializer='zeros', activation='sigmoid'))

        model.compile(loss='categorical_crossentropy')
        model.fit(x_train, y_train, n_epochs=epochs, learning_rate=0.001)

        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print(acc)


if __name__ == '__main__':
    unittest.main()
