from layers.base import Layer
from layers._layers import *

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

        num_samples = input_shape[0]
        input_size = input_shape[1:-1]

        output_size = list()
        for i in range(len(input_size)):
            length = K.compute_output_length(input_size[i], self.kernel_size[i], self.stride[i], self.padding)
            output_size.append(length)

        return (num_samples,) + tuple(output_size) + (self.num_kernel,)

    def forward(self, inputs):
        """

        :param inputs: tuple of (num_inputs, num_rows, num_cols, num_channels)
        :return: outputs of the layer
        """
        assert self.built == True
        assert K.shape(inputs)[1:] == self.input_shape

        outputs, input_conv = K.conv2d(inputs, self.kernel, self.stride, self.padding)

        self.cache['inputs'] = inputs
        self.cache['input_convolutions'] = input_conv
        self.cache['outputs'] = outputs

        return outputs

    def backward(self, delta_y):

        delta_w = K.reshape(
            K.dot(
                delta_y, K.transpose(self.cache['x_conv'])
            ),
            K.shape(self.kernel)
        )
        # delta_x =
        pass