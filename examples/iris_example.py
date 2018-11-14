import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense as KerasDense
from keras.utils import to_categorical

from layers import Dense
from engines.model import Model


epochs = 10000
learning_rate = 0.001

def prepare_datasets():
    X, y = load_iris(True)
    X = preprocessing.normalize(X)
    return train_test_split(X, y, test_size=0.25)


def train_with_sklearn(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier((7, 3), activation="logistic", solver="lbfgs", batch_size=len(X_train), max_iter=10000)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)
    print("MLPClassifier Accuracy score:", accuracy_score(y_test, pred))


def train_with_keras(X_train, X_test, y_train, y_test):
    keras_model = Sequential([
        KerasDense(units=7, activation='sigmoid', input_dim=4),
        KerasDense(units=7, activation='sigmoid'),
        KerasDense(units=3, activation='sigmoid')
    ])
    keras_model.compile(
        optimizer=keras.optimizers.SGD(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    one_hot_labels = to_categorical(y_train, num_classes=3)
    keras_model.fit(X_train, one_hot_labels, epochs=epochs, verbose=0)

    loss, acc = keras_model.evaluate(X_test, to_categorical(y_test, num_classes=3))
    print("Keras loss:", loss)
    print("Keras accuracy:", acc)


def train_with_my_model(X_train, X_test, y_train, y_test):
    layers = [
        Dense(units=7, activation='sigmoid', use_bias=False, input_shape=(4,)),
        Dense(units=7, activation='tanh', use_bias=True),
        Dense(units=3, use_bias=False)
    ]

    model = Model(layers)
    model.compile(loss='categorical_crossentropy')
    one_hot_targets = to_categorical(y_train, num_classes=3)
    model.fit(X_train, one_hot_targets, n_epochs=epochs, learning_rate=learning_rate)

    my_pred = model.predict(X_test)
    print("My accuracy score:", accuracy_score(y_test, my_pred))


def main():
    np.random.seed(1)

    X_train, X_test, y_train, y_test = prepare_datasets()

    # train_with_sklearn(X_train, X_test, y_train, y_test)

    train_with_keras(X_train, X_test, y_train, y_test)

    train_with_my_model(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()