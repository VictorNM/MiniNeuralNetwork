from layers import *
from ops import activations


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=None):
        self.weight = K.random(n_in, n_out)
        self.bias = K.zeros((1, n_out))

        self.activation = None
        self.activation_derivative = None

        self.activation, self.activation_derivative = activations.get(activation)

        self.cache = None
        self.outputs = None

    def forward(self, x):
        self.cache = x
        linear_output = K.dot(x, self.weight) + self.bias
        if self.activation is None:
            self.outputs = linear_output
        else:
            self.outputs = self.activation(linear_output)
        return self.outputs

    def backward(self, delta_o2):
        if self.activation_derivative is not None:
            delta_o1 = delta_o2 * self.activation_derivative(self.outputs)
        else:
            delta_o1 = delta_o2

        delta_weight = K.dot(self.cache.T, delta_o1)
        delta_bias = K.ones(len(delta_o2)).dot(delta_o1)

        return delta_o1, delta_weight, delta_bias