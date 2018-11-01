from layers.base import Layer
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
        self.activation = activations.get(activation)
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

        return outputs

    def backward(self, delta_outputs):
        input_convolutions = self.cache['input_convolutions']
        kernel_convolutions = K.compute_input_convolutions(self.kernel, self.input_shape[1:-1], self.stride, self.padding)


        output_flatten = K.reshape(delta_outputs, ())
        pass

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