from layers.base import Layer
from layers.activation import Activation
from layers._layers import *
from utils.conv_utils import _compute_padding_size
from utils.conv_utils import pad_inputs
from utils.conv_utils import compute_output_shape

# TODO 2: implement backward


class Conv2D(Layer):
    def __init__(self,
                 input_shape,
                 num_kernel,
                 kernel_size,
                 stride=(1,1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros'):

        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2
        assert isinstance(stride, tuple) and len(stride) == 2
        assert padding in {'same', 'valid'}
        assert kernel_initializer, bias_initializer in {'zeros', 'random_normal'}

        self.input_shape = input_shape
        self.num_kernel = int(num_kernel)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = Activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel = None
        self.cache = dict()
        self.built = False

    def build(self):
        num_channel = self.input_shape[-1]
        kernel_shape = (self.num_kernel, ) + self.kernel_size + (num_channel, )
        self.kernel = self.kernel_initializer(kernel_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        kernel_shape = (self.num_kernel, ) + self.kernel_size + (input_shape[-1], )
        return compute_output_shape(input_shape, kernel_shape, self.stride, self.padding)

    def forward(self, inputs):
        """
        Do forward pass
        :param inputs: 4d array of (num_inputs, num_rows, num_cols, num_channels)
        :return: outputs of the layer
        """
        input_shape = K.shape(inputs)
        assert self.built == True
        assert len(input_shape) == 4
        assert input_shape[1:] == self.input_shape

        outputs = self._conv2d(inputs)

        if self.activation is not None:
            outputs = self.activation.forward(outputs)

        return outputs

    def backward(self, delta_outputs):

        if self.activation is not None:
            delta_outputs = self.activation.backward(delta_outputs)['delta_inputs']

        delta_kernels = self._compute_delta_kernels(delta_outputs)
        delta_inputs = self._compute_delta_inputs(delta_outputs)

        return {
            'delta_kernels': delta_kernels,
            'delta_inputs': delta_inputs
        }

    def update_params(self, adjustment, learning_rate):
        self.kernel -= adjustment['delta_kernels'] * learning_rate

    def _conv2d(self, inputs):
        input_shape = K.shape(inputs)
        num_input = input_shape[0]

        output_shape = self.compute_output_shape(input_shape)
        output_size = output_shape[1:-1]

        # pad input
        padded_inputs = pad_inputs(inputs, self.kernel_size, self.padding)

        # compute the convolution matrices of inputs
        input_convolutions = K.compute_input_convolutions(padded_inputs, self.kernel_size, output_size, self.stride)

        # compute outputs
        outputs = K.empty(output_shape)
        for i in range(num_input):
            outputs[i] = K.compute_multi_channel_output(input_convolutions[i], self.kernel, output_size)

        # save cache for backward
        self.cache['inputs'] = inputs
        self.cache['input_convolutions'] = input_convolutions
        self.cache['outputs'] = outputs

        return outputs

    def _compute_delta_kernels(self, delta_outputs):
        output_shape = K.shape(delta_outputs)
        output_flatten_shape = (output_shape[0], output_shape[1] * output_shape[2], output_shape[3])
        output_flatten = K.reshape(delta_outputs, output_flatten_shape)
        input_convolutions = self.cache['input_convolutions']

        num_input = output_shape[0]
        kernels_shape = (self.num_kernel, ) + self.kernel_size + (self.input_shape[-1], )
        delta_kernels = K.empty(kernels_shape)
        for i in range(num_input):
            delta_kernels += K.compute_delta_kernels(output_flatten[i], input_convolutions[i], self.kernel_size)

        return delta_kernels

    def _compute_delta_inputs(self, delta_outputs):
        outputs_shape = K.shape(delta_outputs)
        outputs_flatten_shape = (outputs_shape[0], outputs_shape[1] * outputs_shape[2], outputs_shape[3])
        outputs_flatten = K.reshape(delta_outputs, outputs_flatten_shape)

        num_input = outputs_shape[0]
        input_size = self.input_shape[0:-1]
        inputs_shape = (num_input, ) + self.input_shape
        kernel_convolutions = K.compute_kernel_convolutions(self.kernel, input_size, outputs_shape[1:-1])

        delta_inputs = K.empty(inputs_shape)
        for i in range(num_input):
            delta_inputs[i] = K.compute_delta_input(outputs_flatten[i], kernel_convolutions, self.input_shape)

        return delta_inputs
