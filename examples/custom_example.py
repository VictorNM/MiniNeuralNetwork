import numpy as np

import keras

from engines.model import Model
from layers import Dense, Conv2D, Flatten
from utils.train_utils import to_categorical
from sklearn.metrics import accuracy_score


input_shape = (5, 5, 1)
num_classes = 2
num_kernels = 1
epochs = 10


def prepare_data():
    x = np.array([
        # 2 lines
        np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]).reshape(input_shape),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(input_shape),
        np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]).reshape(input_shape),
        # 1 line
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(input_shape),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(input_shape),
        np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]).reshape(input_shape),
        # test
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(input_shape),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(input_shape)
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    x_train, x_test = x[:6], x[6:]
    y_train, y_test = y[:6], y[6:]

    return x_train, y_train, x_test, y_test


def train_with_keras(x_train, y_train, x_test, y_test):
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(num_kernels, kernel_size=(3, 3),
                                  kernel_initializer='zeros',
                                  use_bias=False,
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_classes,
                                 kernel_initializer='zeros',
                                 bias_initializer='zeros',
                                 activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.1),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=len(x_train),
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def train_with_my_model(x_train, y_train, x_test, y_test):
    y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    model = Model()
    model.add(Conv2D(
        num_kernel=num_kernels,
        kernel_size=(3,3),
        kernel_initializer='zeros',
        use_bias=False,
        activation='relu',
        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(
        units=num_classes,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        activation='sigmoid'))

    model.compile(loss='categorical_crossentropy')
    model.fit(x_train, y_train, n_epochs=epochs, learning_rate=0.1)

    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print(acc)


def main():
    x_train, y_train, x_test, y_test = prepare_data()
    train_with_keras(x_train, y_train, x_test, y_test)
    train_with_my_model(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()
