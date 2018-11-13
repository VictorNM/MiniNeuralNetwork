from layers.base import Layer
from layers._layers import *

class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

        self.cache = dict()

    def forward(self, inputs):
        inputs_shape = K.shape(inputs)
        self.cache['inputs_shape'] = inputs_shape

        outputs_shape = self.compute_outputs_shape(inputs_shape)
        return K.reshape(inputs, outputs_shape)

    def backward(self, delta_outputs, learning_rate):
        return K.reshape(delta_outputs, self.cache['inputs_shape'])

    def compute_outputs_shape(self, inputs_shape):
        assert len(inputs_shape) >= 2

        input_flatten_shape = 1
        input_shape = inputs_shape[1:]
        for i in range(len(input_shape)):
            input_flatten_shape *= input_shape[i]

        inputs_flatten_shape = (inputs_shape[0], ) + (input_flatten_shape, )

        return inputs_flatten_shape
