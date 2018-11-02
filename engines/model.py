import backend as K
from layers.base import Layer
from ops import losses


class Model:
    def __init__(self, layers=None):
        if layers is not None and not isinstance(layers, list):
            raise RuntimeError("_layers must be an list of Layer")

        self._layers = None
        self._is_compiled = False

        if isinstance(layers, list):
            for layer in layers:
                self.add(layer)

    def fit(self, X, y, n_epochs=2000, learning_rate=0.001):
        if not self._is_compiled:
            raise RuntimeError("You have to compile the model before use it!")

        for epoch in range(n_epochs):
            y_hat = self._do_forward(X)
            self.training_outputs = y_hat

            loss = self._loss_function(y, y_hat)
            if epoch % 1000 == 0:
                print("Loss at epochs %d: %f" %(epoch, loss))

            adjustments = self._do_backward(y)
            self.update_params(adjustments, learning_rate)

    def predict(self, X):
        o = self._do_forward(X)

        return K.argmax(o, axis=1)

    def compile(self, loss=None):
        for layer in self._layers:
            layer.build()

        self._loss_function = losses.get(loss)
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

    def _do_backward(self, y):
        adjustments = list()

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer is self._layers[-1]:
                delta_y = self.training_outputs - y
            else:
                delta_y = backward_output['delta_inputs']

            backward_output = layer.backward(delta_y)
            adjustments.insert(0, backward_output)

        return adjustments

    def update_params(self, adjustments, learning_rate):
        for i, layer in enumerate(self._layers):
            layer.update_params(adjustments[i], learning_rate)