from layers.base import Layer
from layers._layers import *


class Dense(Layer):
    def __init__(self, n_in, n_out,
                 activation=None,
                 use_bias=True):
        
        self.weight = initializers.random_normal((n_in, n_out))
        self.use_bias = use_bias
        if use_bias:
            self.bias = initializers.zeros((1, n_out))

        self.activation, self.activation_derivative = activations.get(activation)

        self.cache = None
        self.outputs = None

    def forward(self, x):
        self.cache = x
        linear_output = K.dot(x, self.weight)

        if self.use_bias:
            linear_output += self.bias

        self.outputs = self.activation(linear_output)
        return self.outputs

    def backward(self, delta_y):
        e = delta_y * self.activation_derivative(self.outputs)
        delta_weight = K.dot(self.cache.T, e)
        delta_bias = K.ones(len(delta_y)).dot(e)

        adjustment = {
            'delta_weight': delta_weight,
            'delta_bias': delta_bias
        }

        return e, adjustment

    def update_params(self, adjustment, learning_rate):
        self.weight -= adjustment['delta_weight'] * learning_rate
        if self.use_bias:
            self.bias -= adjustment['delta_bias'] * learning_rate