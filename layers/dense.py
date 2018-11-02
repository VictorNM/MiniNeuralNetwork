from layers.base import Layer
from layers.activation import Activation
from layers._layers import *


class Dense(Layer):
    def __init__(self, n_in, n_out,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros'):

        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if use_bias:
            self.bias = initializers.zeros((1, n_out))

        self.activation = Activation(activation)

        self.cache = dict()
        self.built = False

    def build(self):
        self.kernels = self.kernel_initializer((self.n_in, self.n_out))
        if self.use_bias:
            self.bias = self.bias_initializer((1, self.n_out))

        self.built = True

    def forward(self, x):
        self.cache['inputs'] = x
        outputs = K.dot(x, self.kernels)

        if self.use_bias:
            outputs += self.bias

        outputs = self.activation.forward(outputs)

        return outputs

    def backward(self, delta_outputs):
        inputs = self.cache['inputs']

        delta_activation = self.activation.backward(delta_outputs)['delta_inputs']
        delta_weight = K.dot(K.transpose(inputs), delta_activation)
        delta_bias = K.ones(len(delta_outputs)).dot(delta_activation)
        delta_input = K.dot(delta_activation, K.transpose(self.kernels))

        backward_output = {
            'delta_inputs': delta_input,
            'delta_kernels': delta_weight,
            'delta_bias': delta_bias
        }

        return backward_output

    def update_params(self, adjustment, learning_rate):
        self.kernels -= adjustment['delta_kernels'] * learning_rate
        if self.use_bias:
            self.bias -= adjustment['delta_bias'] * learning_rate

    def compute_output_shape(self, input_shape):
        pass