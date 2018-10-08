from layers.base import *
from ops.activations import *


class Dense(Layer):
    def __init__(self, n_in, n_out, activation='sigmoid'):
        self.weight = np.random.rand(n_in, n_out) * 0.01
        self.activation = None
        self.activation_derivative = None
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

        self.cache = None
        self.outputs = None

    def forward(self, x):
        self.cache = x
        linear_output = np.dot(x, self.weight)
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

        delta_weight = np.dot(self.cache.T, e)

        return e, delta_weight