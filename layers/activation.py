from layers.base import Layer
from layers._layers import *


class Activation(Layer):
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activations.get(activation)
        self.activation_derivative = activations.get_derivative(activation)

        self.cache = dict()

    def forward(self, inputs):
        outputs = self.activation(inputs)
        self.cache['inputs'] = inputs
        self.cache['outputs'] = outputs
        return outputs

    def backward(self, delta_outputs):
        inputs = self.cache['inputs']
        delta_inputs = delta_outputs * self.activation_derivative(inputs)

        return delta_inputs

    def compute_outputs_shape(self, inputs_shape):
        return inputs_shape
