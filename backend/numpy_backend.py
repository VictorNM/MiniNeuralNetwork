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

def _pad(matrix3d, padding_size, mode='constant'):
    assert len(shape(matrix3d)) == 3
    assert len(padding_size) == 2

    padding_size = ((padding_size[0], ), (padding_size[1], ), (0, ))
    return np.pad(matrix3d, padding_size, mode)


def _rotate180(matrix, axis=(0,1)):
    return np.rot90(matrix, 2, axis)


def _compute_padding_length(kernel_length, padding):
    if padding == 'valid':
        return 0
    if padding == 'same':
        return kernel_length // 2


def _compute_padding_size(kernel_size, padding):
    padding_size = list()
    for i in range(len(kernel_size)):
        padding_size.append(_compute_padding_length(kernel_size[i], padding))

    return tuple(padding_size)


def compute_output_length(input_length, kernel_length, stride=1, padding='valid'):
    input_length = input_length + 2 * _compute_padding_length(kernel_length, padding)
    return (input_length - kernel_length) // stride + 1


def compute_input_convolution(inputs, kernel_size, stride=(1, 1), padding='valid'):
    """
    compute input convolution for a single input with single or multi channel
    :param inputs: 3d matrix of shape (num_row, num_col, num_channel)
    :param kernel_size: size of a 2d kernel (num_row, num_col)
    :return: input convolution with the same channel of input
    """

    padding_size = _compute_padding_size(kernel_size, padding)
    padded_inputs = _pad(inputs, padding_size)

    input_shape = shape(inputs)
    num_channel = input_shape[-1]
    input_size = input_shape[:-1]

    n_output_row = compute_output_length(input_size[0], kernel_size[0], stride[0], padding)
    n_output_col = compute_output_length(input_size[1], kernel_size[1], stride[1], padding)
    n_input_conv_row = kernel_size[0] * kernel_size[1]
    n_input_conv_col = n_output_row * n_output_col

    input_convolution = empty((n_input_conv_row, n_input_conv_col, num_channel))
    for i in range(n_output_row):
        for j in range(n_output_col):
            marked_area = padded_inputs[i:i+kernel_size[0], j:j+kernel_size[1], :]
            row_conv = reshape(_rotate180(marked_area), (n_input_conv_row, num_channel))
            input_convolution[:, i*n_output_col+j, :] = row_conv

    return input_convolution


def conv2d_one_input(one_input, kernels, stride=(1, 1), padding='valid'):
    """
    Compute the multi-channel output by applying all kernel on a single input
    :param one_input: 3d matrix (num_row, num_col, num_channel)
    :param kernels: 4d matrix (num_kernel, num_row, num_col, num_channel)
    :param stride:
    :param padding:
    :return: output of the convolution
    """
    input_shape = shape(one_input)
    kernel_shape = shape(kernels)

    num_input_channel = input_shape[-1]
    num_output_channel = kernel_shape[0]    # the number of output channel == the number of kernel
    kernel_size = kernel_shape[1:-1]
    input_conv = compute_input_convolution(one_input, kernel_size, stride, padding)

    n_ouput_row = compute_output_length(input_shape[0], kernel_size[0], stride[0], padding)
    n_output_col = compute_output_length(input_shape[1], kernel_size[1], stride[1], padding)

    output_size = (n_ouput_row, n_output_col)
    output_shape = output_size + (num_output_channel, )

    outputs = empty(output_shape)
    for k in range(num_output_channel):
        one_output_channel = zeros(output_size)
        for c in range(num_input_channel):
            # Y = X * W = Reshape(W_flat x X_conv)
            one_output_channel += reshape(dot(flat(kernels[:,:,:,c]), input_conv[:,:,c]), output_size)

        outputs[:,:,k] = one_output_channel

    return outputs, input_conv


def conv2d(inputs, kernels, stride=(1, 1), padding='valid'):
    """
    Compute convolutional outputs for many inputs
    :param inputs: 4d matrix (num_input, num_row, num_col, num_channel)
    :param kernels: 4d matrix (num_kernel, num_row, num_col, num_channel)
    :param stride: 2d matrix (vertical_stride, horizontal_stride)
    :param padding: 'same' or 'valid'
    :return: 4d matrix (num_output, num_row, num_col, num_channel)
    """
    input_shape = shape(inputs)
    kernel_shape = shape(kernels)

    assert len(input_shape) == 4
    assert len(kernel_shape) == 4
    assert input_shape[-1] == kernel_shape[-1]

    num_input = input_shape[0]
    num_kernel = kernel_shape[0]
    outputs = list()

    input_convs = list()

    for i in range(num_input):
        output, input_conv = conv2d_one_input(inputs[i], kernels, stride, padding)
        outputs.append(output)
        input_convs.append(input_conv)

    return np.array(outputs), np.array(input_convs)


def calculate_kernel_convolution(kernel, input_size, stride=(1, 1), padding='valid'):

    kernel_shape = shape(kernel)

    assert len(kernel_shape) == 3
    assert len(input_size) == 2

    num_channel = kernel_shape[-1]

    n_output_row = compute_output_length(input_size[0], kernel_shape[0], stride[0], padding)
    n_output_col = compute_output_length(input_size[1], kernel_shape[1], stride[1], padding)
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


def calculate_kernel_convolutions(kernels, input_size, stride=(1, 1), padding='valid'):
    kernel_convolutions = list()
    for i in range(len(kernels)):
        kernel_convolution = calculate_kernel_convolution(kernels[i], input_size, stride, padding)
        kernel_convolutions.append(kernel_convolution)

    return np.array(kernel_convolutions)