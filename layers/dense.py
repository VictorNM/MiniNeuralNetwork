from layers.base import *
from ops import activations


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=None):
        self.weight = np.random.rand(n_in, n_out) * 0.01
        self.bias = np.zeros((1, n_out))

        self.activation = None
        self.activation_derivative = None

        self.activation, self.activation_derivative = activations.get(activation)

        self.cache = None
        self.outputs = None

    def forward(self, x):
        self.cache = x
        linear_output = np.dot(x, self.weight) + self.bias
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

        delta_weight = np.dot(self.cache.T, delta_o1)
        delta_bias = np.ones(len(delta_o2)).dot(delta_o1)

        return delta_o1, delta_weight, delta_bias