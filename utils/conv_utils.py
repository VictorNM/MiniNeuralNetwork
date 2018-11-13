import backend as K


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


def pad_inputs(inputs, kernel_size, padding='valid'):
    padding_size = _compute_padding_size(kernel_size, padding)
    padding_size = ((0,), (padding_size[0],), (padding_size[1],), (0,))
    return K.pad(inputs, padding_size)


def _compute_output_length(input_length, kernel_length, stride=1, padding='valid'):
    input_length = input_length + 2 * _compute_padding_length(kernel_length, padding)
    return (input_length - kernel_length) // stride + 1


def compute_output_shape(input_shape, kernel_shape, stride=(1,1), padding='valid'):

    assert len(input_shape) == 4
    assert len(kernel_shape) == 4

    num_input = input_shape[0]
    num_kernel = kernel_shape[0]
    input_size = input_shape[1:-1]
    kernel_size = kernel_shape[1:-1]

    output_size = list()
    for i in range(len(input_size)):
        output_size.append(_compute_output_length(input_size[i], kernel_size[i], stride[i], padding))

    return (num_input, ) + tuple(output_size) + (num_kernel, )