import logging

import backend as K
from layers.base import Layer
from ops import losses


class Model:
    def __init__(self, layers=None):
        if layers is not None and not isinstance(layers, list):
            raise RuntimeError("_layers must be an list of Layer")

        self._layers = None
        self._is_compiled = False
        self._built = False

        if isinstance(layers, list):
            for layer in layers:
                self.add(layer)

    def fit(self, x, y, n_epochs=2000, learning_rate=0.001):
        if not self._is_compiled:
            raise RuntimeError("You have to compile the model before use it!")

        if not self._built:
            self.build(K.shape(x))

        for epoch in range(n_epochs):
            y_hat = self._do_forward(x)
            self.training_outputs = y_hat

            self.loss = self._loss_function(y, y_hat)
            if epoch % 1 == 0:
                print("Loss at epochs %d: %f" %(epoch, self.loss))

            self._do_backward(y, learning_rate)

    def predict(self, X):
        o = self._do_forward(X)

        return K.argmax(o, axis=1)

    def build(self, input_shape):
        shape = input_shape
        for layer in self._layers:
            layer.build(shape)
            shape = layer.compute_outputs_shape(shape)

        self._built = True

    def compile(self, loss=None):
        self._loss_function = losses.get(loss)
        self._loss_function_derivative = losses.get_derivative(loss)
        self._is_compiled = True

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise RuntimeError("layers must be an list of Layer")

        if self._layers is None:
            self._layers = list()

        self._layers.append(layer)

    def _do_forward(self, X):
        o = X
        for layer in self._layers:
            o = layer.forward(o)

        return o

    def _do_backward(self, y, learning_rate):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer is self._layers[-1]:
                delta_y = self._loss_function_derivative(y, self.training_outputs)
            else:
                delta_y = delta_x

            delta_x = layer.backward(delta_y, learning_rate)