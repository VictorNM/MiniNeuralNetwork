import numpy as np
import collections


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)


def sum(x, axis=-1):
    return np.sum(x, axis, keepdims=True)


def dot(x, y):
    return np.dot(x, y)


def argmax(x, axis=-1):
    return np.argmax(x, axis)


def square(x):
    return np.square(x)


def abs(x):
    return np.abs(x)


def exp(x):
    return np.exp(x)


def random(x, y):
    return np.random.rand(x, y)


def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


# === activations ===


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    return x * (1.0 - x)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1.0 - np.square(x)


def relu(x):
    return np.maximum(x, 0)


def d_relu(x):
    return 1.0 * (x > 0)


# TODO: implement softmax function
def softmax(x):
    pass

# === loss ===


def mean_square_error(y, y_hat):
    if isinstance(y, (collections.Sequence, np.ndarray)):
        # at least 2d array
        if isinstance(y[0], (collections.Sequence, np.ndarray)):
            return np.mean(1.0 / 2.0 * np.sum(np.square(y_hat - y), axis=1))
        # 1d array
        else:
            return np.mean(1.0 / 2.0 * np.square(y_hat - y))
    # scalar number
    else:
        return 1.0 / 2.0 * np.square(y_hat - y)


def categorical_crossentropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-12, 1. - 1e-12)
    ce = -np.mean(np.log(y_hat) * y)
    return ce
