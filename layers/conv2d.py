from layers.base import Layer
from layers._layers import *

# TODO 1: implement forward
# TODO 2: implement backward
# TODO 3: implement multi-chanel input
# TODO 4: implement multi-filter


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

        self.input_shape = input_shape
        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.outputs = None
        self.cache = dict()
        self.built = False

    def build(self):
        input_dim = self.input_shape[-1]
        kernel_shape = (self.num_kernel, ) + self.kernel_size + (input_dim, )
        self.kernel = initializers.random_normal(kernel_shape)
        self.built = True

    def compute_output_shape(self, input_shape):

        num_samples = input_shape[0]
        input_size = input_shape[1:-1]

        if self.padding == 'same':
            output_size = input_size

        if self.padding == 'valid':
            output_row = (input_size[0] - self.kernel_size[0]) // self.stride[0] + 1
            output_col = (input_size[1] - self.kernel_size[1]) // self.stride[1] + 1
            output_size = (output_row, output_col)

        return (num_samples,) + tuple(output_size) + (self.num_kernel,)

    def forward(self, inputs):
        """

        :param inputs: tuple of (num_inputs, num_rows, num_cols, num_channels)
        :return: outputs of the layer
        """
        assert self.built == True
        return K.conv2d(inputs, self.kernel, self.stride, self.padding)

    def backward(self, delta_y):
        delta_w = K.reshape(
            K.dot(
                delta_y, K.transpose(self.cache['x_conv'])
            ),
            K.shape(self.kernel)
        )
        # delta_x =
        pass