from layers.base import Layer
from layers._layers import *

# TODO: implement this class


class Conv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride=(1,1),
                 padding='valid',
                 activation=None,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros'):

        self.kernel = initializers.random_normal(kernel_size)
        pass

    def forward(self, x):
        x_conv = K.im2col(x, self.kernel.shape)
        o = K.dot(self.kernel.flatten(), x_conv).reshape((2,2))
        pass

    def backward(self):
        pass