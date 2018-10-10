from layers import *
from ops import activations


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=None, use_bias=True):
        self.weight = K.random(n_in, n_out)
        self.use_bias = use_bias
        if use_bias:
            self.bias = K.zeros((1, n_out))

        self.activation = None
        self.activation_derivative = None

        self.activation, self.activation_derivative = activations.get(activation)

        self.cache = None
        self.outputs = None

    def forward(self, x):
        self.cache = x
        linear_output = K.dot(x, self.weight)

        if self.use_bias:
            linear_output += self.bias

        if self.activation is None:
            self.outputs = linear_output
        else:
            self.outputs = self.activation(linear_output)
        return self.outputs

    def backward(self, delta_y):
        if self.activation_derivative is not None:
            e = delta_y * self.activation_derivative(self.outputs)
        else:
            e = delta_y

        delta_weight = K.dot(self.cache.T, e)
        delta_bias = K.ones(len(delta_y)).dot(e)

        return e, delta_weight, delta_bias

    def update_params(self, adjustment, learning_rate):
        self.weight -= adjustment['delta_weight'] * learning_rate
        if self.use_bias:
            self.bias -= adjustment['delta_bias'] * learning_rate