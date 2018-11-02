from layers.base import Layer
from layers._layers import *


class Activation(Layer):
    def __init__(self, activation):
        self.activation = activations.get(activation)
        self.activation_derivative = activations.get_derivative(activation)

        self.cache = dict()

    def forward(self, inputs):
        outputs = self.activation(inputs)
        self.cache['inputs'] = inputs
        self.cache['outputs'] = outputs
        return outputs

    def backward(self, delta_outputs):
        if activations.serialize(self.activation) == 'relu':
            inputs = self.cache['inputs']
            grad = self.activation_derivative(inputs)
            delta_inputs = delta_outputs * self.activation_derivative(inputs)
        else:
            outputs = self.cache['outputs']
            delta_inputs = delta_outputs * self.activation_derivative(outputs)
        return {
            'delta_inputs' : delta_inputs
        }

    def compute_output_shape(self, input_shape):
        return input_shape


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__(activations='sigmoid')

    def forward(self, inputs):
        return self.activation(inputs)