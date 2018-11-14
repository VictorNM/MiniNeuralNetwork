import numpy as np
import collections


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)


def empty(shape, dtype=float):
    return np.empty(shape, dtype)


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


def transpose(x):
    return np.transpose(x)


def flat(x):
    return x.flatten()


def shape(x):
    return np.shape(x)


def reshape(x, new_shape):
    return np.reshape(x, new_shape)

# === activations ===


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1.0 - np.square(tanh(x))


def relu(x):
    return np.maximum(x, 0)


def d_relu(x):
    return 1.0 * (x >= 0)


def softmax(x):
    normalize_x = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = exp(normalize_x)
    return exp_x / sum(exp_x, axis=-1)

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


def mean_square_error_derivative(y, y_hat):
    return y_hat - y


def categorical_crossentropy(y, y_hat, epsilon=1e-7):
    assert shape(y) == shape(y_hat)
    p = softmax(y_hat)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.mean(np.sum(y * -np.log(p), axis=-1, keepdims=True))


def categorical_crossentropy_derivative(y, y_hat):
    p = softmax(y_hat)
    return p - y

# === convolution ====


def pad(inputs, padding_size, mode='constant'):
    return np.pad(inputs, padding_size, mode)


def _rotate180(matrix, axis=(0,1)):
    return np.rot90(matrix, 2, axis)


def _compute_input_convolution(inputs, kernel_size, output_size, stride=(1, 1)):
    """
    compute input convolution for a single input with single or multi channel
    NOTE: this function assumes that inputs are already padded
    :param inputs: 3d matrix of shape (num_row, num_col, num_channel)
    :param kernel_size: size of a 2d kernels (num_row, num_col)
    :return: input convolution with the same channel of input
    """

    input_shape = shape(inputs)
    num_channel = input_shape[-1]

    n_output_row, n_output_col = output_size

    n_input_convolution_row = kernel_size[0] * kernel_size[1]
    n_input_convolution_col = n_output_row * n_output_col
    input_convolution_shape = (n_input_convolution_row, n_input_convolution_col, num_channel)

    input_convolution = empty(input_convolution_shape)
    for i in range(n_output_row):
        for j in range(n_output_col):
            start_row, start_col = i*stride[0], j*stride[1]
            end_row, end_col = start_row + kernel_size[0], start_col + kernel_size[1]
            marked_area = inputs[start_row:end_row, start_col:end_col, :]
            row_conv = reshape(_rotate180(marked_area), (n_input_convolution_row, num_channel))
            input_convolution[:, i*n_output_col+j, :] = row_conv

    return input_convolution


def compute_input_convolutions(inputs, kernel_size, output_size, stride=(1, 1)):

    assert len(kernel_size) == 2

    input_shape = shape(inputs)
    num_input = input_shape[0]

    input_convolutions = list()

    for i in range(num_input):
        input_convolutions.append(_compute_input_convolution(inputs[i], kernel_size, output_size, stride))

    return np.array(input_convolutions)


def _compute_kernel_convolution(kernel, input_size, output_size, stride=(1, 1)):

    kernel_shape = shape(kernel)

    assert len(kernel_shape) == 3
    assert len(input_size) == 2

    num_channel = kernel_shape[-1]

    n_output_row, n_output_col = output_size
    n_kernel_conv_row = n_output_row * n_output_col
    n_kernel_conv_col = input_size[0] * input_size[1]

    kernel = _rotate180(kernel)
    n_kernel_conv_shape = ((n_kernel_conv_row, n_kernel_conv_col) + (num_channel, ))

    kernel_convolution = zeros(n_kernel_conv_shape)
    for i in range(n_output_row):
        for j in range(n_output_col):
            x = zeros(input_size + (num_channel,))
            x[i:i + kernel.shape[0], j:j + kernel.shape[1]] = kernel
            kernel_convolution[i*n_output_col+j, :] = reshape(x, (n_kernel_conv_col, num_channel))

    return kernel_convolution


def compute_kernel_convolutions(kernels, input_size, output_size, stride=(1, 1)):
    kernel_convolutions = list()
    for i in range(len(kernels)):
        kernel_convolution = _compute_kernel_convolution(kernels[i], input_size, output_size, stride)
        kernel_convolutions.append(kernel_convolution)

    return np.array(kernel_convolutions)


def _compute_one_channel_output(input_convolution, kernel, output_size):
    kernel_shape = shape(kernel)
    kernel_size = kernel_shape[:-1]
    num_channel = kernel_shape[-1]
    kernel_flat = reshape(kernel, (kernel_size[0] * kernel_size[1], num_channel))

    output = zeros(output_size)
    for i in range(num_channel):
        output += reshape(dot(kernel_flat[:, i], input_convolution[:, :, i]), output_size)

    return output


def compute_multi_channel_output(input_convolution, kernels, output_size):
    kernel_shape = shape(kernels)
    num_kernel = kernel_shape[0]

    output_shape = output_size + (num_kernel, )
    output = empty(output_shape)

    for i in range(num_kernel):
        one_channel_output = _compute_one_channel_output(input_convolution, kernels[i], output_size)
        output[:, :, i] = one_channel_output

    return output


def _compute_delta_kernel(output_flatten, input_convolution, kernel_size):
    output_flatten_shape = shape(output_flatten)
    input_convolution_shape = shape(input_convolution)

    assert len(output_flatten_shape) == 1

    num_channel = input_convolution_shape[-1]
    kernel_shape = kernel_size + (num_channel, )
    delta_kernel = empty(kernel_shape)
    for i in range(num_channel):
        delta_kernel[:, :, i] = reshape(dot(output_flatten, transpose(input_convolution[:, :, i])), kernel_size)

    return delta_kernel


def compute_delta_kernels(output_flatten, input_convolution, kernel_size):
    output_flatten_shape = shape(output_flatten)
    input_convolution_shape = shape(input_convolution)

    assert len(output_flatten_shape) == 2     # (flatten_size, num_output_channel)
    assert len(input_convolution_shape) == 3  # (num_row, num_col, num_input_channel)

    num_kernel =  output_flatten_shape[-1]
    num_channel = input_convolution_shape[-1]

    kernels_shape = (num_kernel, ) + kernel_size + (num_channel, )

    delta_kernels = empty(kernels_shape)
    for i in range(num_kernel):
        delta_kernels[i] = _compute_delta_kernel(output_flatten[:, i], input_convolution, kernel_size)

    return delta_kernels


def compute_delta_input(output_flatten, kernel_convolutions, input_shape):
    output_flatten_shape = shape(output_flatten)
    kernel_convolutions_shape = shape(kernel_convolutions)

    assert len(output_flatten_shape) == 2
    assert len(kernel_convolutions_shape) == 4

    num_kernels = kernel_convolutions_shape[0]
    num_channels = input_shape[-1]

    delta_input = zeros(input_shape)
    for i in range(num_kernels):
        for j in range(num_channels):
            delta_input[:, :, j] += reshape(
                transpose(dot(output_flatten[:, i], kernel_convolutions[i, :, :, j])),
                input_shape[0:-1])

    return delta_input
