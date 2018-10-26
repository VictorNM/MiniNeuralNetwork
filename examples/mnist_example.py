
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D

from keras import backend as K

# from layers.dense import Dense

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols, num_channels = 28, 28, 1
input_shape = (img_rows, img_cols, num_channels)


def prepare_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reduce sample
    num_samples = 1000
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]

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

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])       # 0.36969225867455824
    print('Test accuracy:', score[1])   # 0.9074


def test_with_my_model(x_train, y_train, x_test, y_test):
    pass


def main():
    x_train, y_train, x_test, y_test = prepare_data()
    # train_with_keras(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()