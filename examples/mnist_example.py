
import keras
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

import numpy as np

from engines.model import Model
from layers import Dense, Conv2D, Flatten
from utils.train_utils import to_categorical

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols, num_channels = 28, 28, 1
input_shape = (img_rows, img_cols, num_channels)


def prepare_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reduce sample
    num_train_samples = 100
    x_train = x_train[:num_train_samples]
    y_train = y_train[:num_train_samples]

    num_test_samples = int(0.3 * num_train_samples)
    x_test = x_test[:num_test_samples]
    y_test = y_test[:num_test_samples]

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, y_train, x_test, y_test


def train_with_keras(x_train, y_train, x_test, y_test):
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(1, kernel_size=(3, 3),
                                  kernel_initializer='zeros',
                                  use_bias=False,
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_classes,
                                 kernel_initializer='zeros',
                                 activation='sigmoid'))
    model.add(keras.layers.Activation(activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])       # 0.36969225867455824
    print('Test accuracy:', score[1])   # 0.9074


def train_with_my_model(x_train, y_train, x_test, y_test):
    y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    model = Model()
    model.add(Conv2D(num_kernel=1, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=num_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy')
    model.fit(x_train, y_train, n_epochs=epochs, learning_rate=0.01)

    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print(acc)


def main():
    x_train, y_train, x_test, y_test = prepare_data()
    train_with_keras(x_train, y_train, x_test, y_test)
    train_with_my_model(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()