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


def random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=stddev, size=shape)


def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


def flat(x):
    return x.flatten()

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


def categorical_crossentropy(y, y_hat, epsilon=1e-12):
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    ce = -np.mean(np.log(y_hat) * y)
    return ce


# === convolution ====

def _pad(x, padding_size, mode='constant'):
    return np.pad(x, padding_size, mode)


def im2col(x, w_shape, stride=1, padding='valid'):

    '''

    :param x: input image
    :param w_shape: kernel's shape
    :param stride: value of stride
    :param padding: 'valid' / 'same'
    :return: x_conv
    '''

    if padding == 'same':
        if stride != 1:
            raise RuntimeError("Padding 'same' only accept stride=1")
        else:
            padding_size = (int(w_shape[0] / 2), int(w_shape[1] / 2))
            x = _pad(x, padding_size, 'constant')

    n_row = w_shape[0] * w_shape[1]
    n_vertical_step = int((x.shape[0] - w_shape[0]) / stride) + 1
    n_horizontal_step = int((x.shape[1] - w_shape[1]) / stride) + 1
    n_col = n_horizontal_step * n_vertical_step

    x_conv = zeros((n_row, n_col))
    for i in range(n_vertical_step):
        for j in range(n_horizontal_step):
            x_conv[:, i*n_horizontal_step+j] = np.copy(x[i:i+w_shape[0], j:j+w_shape[1]].flatten()[::-1])

    return x_conv


def kernel2row(x, w_shape, stride=1, padding=None):
    pass