import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from layers.dense import Dense
from engines.model import Model


def prepare_datasets():
    X, y = load_iris(True)
    X = preprocessing.normalize(X)
    return train_test_split(X, y, test_size=0.25)


def main():
    np.random.seed(7)

    X_train, X_test, y_train, y_test = prepare_datasets()
    mlp = MLPClassifier((7,3), activation="logistic", solver="lbfgs", batch_size=len(X_train), max_iter=10000)
    mlp.fit(X_train, y_train)

    pred = mlp.predict(X_test)
    print("Accuracy score:", accuracy_score(y_test, pred))

    layers = [
        Dense(4, 7, 'sigmoid'),
        Dense(7, 7, 'sigmoid'),
        Dense(7, 3, 'sigmoid')
    ]

    model = Model(layers)
    one_hot_targets = np.eye(3)[y_train]
    model.fit(X_train, one_hot_targets, n_epochs=20000, learning_rate=0.02)

    my_pred = model.predict(X_test)
    print("My accuracy score:", accuracy_score(y_test, my_pred))

if __name__ == '__main__':
    main()