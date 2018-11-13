from layers.base import Layer
from layers.activation import Activation
from layers._layers import *


class Dense(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros',
                 input_shape=None):

        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = None

        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if use_bias:
            self.bias = initializers.zeros((1, units))

        self.activation = Activation(activation)

        self.cache = dict()
        self.built = False

    def build(self, inputs_shape):
        assert len(inputs_shape) >= 2
        self.input_shape = inputs_shape[1:]

        input_shape_flatten = 1
        for i in range(len(self.input_shape)):
            input_shape_flatten *= self.input_shape[i]

        self.kernels = self.kernel_initializer((input_shape_flatten, self.units))
        if self.use_bias:
            self.bias = self.bias_initializer((1, self.units))
        else:
            self.bias = None

        self.built = True

    def forward(self, inputs):
        self.cache['inputs'] = inputs
        outputs = K.dot(inputs, self.kernels)

        if self.use_bias:
            outputs += self.bias

        if self.activation is not None:
            outputs = self.activation.forward(outputs)

        return outputs

    def backward(self, delta_outputs, learning_rate):
        inputs = self.cache['inputs']

        if self.activation is not None:
            delta_activation = self.activation.backward(delta_outputs)
        else:
            delta_activation = delta_outputs

        delta_weight = K.dot(K.transpose(inputs), delta_activation)
        self.kernels -= delta_weight * learning_rate

        if self.bias is not None:
            delta_bias = K.ones(len(delta_outputs)).dot(delta_activation)
            self.bias -= delta_bias * learning_rate

        delta_inputs = K.dot(delta_activation, K.transpose(self.kernels))

        return delta_inputs

    def compute_outputs_shape(self, inputs_shape):
        assert len(inputs_shape) >= 2
        assert inputs_shape[-1]

        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.units

        return tuple(outputs_shape)
