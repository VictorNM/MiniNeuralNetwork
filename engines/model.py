import backend as K
from ops import losses


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.is_compiled = False

    def fit(self, X, y, n_epochs=2000, learning_rate=0.001):
        if not self.is_compiled:
            raise RuntimeError("The model was never compiled!")

        for epoch in range(n_epochs):
            y_hat = self.do_forward(X)

            loss = self.loss_function(y, y_hat)
            if epoch % 1000 == 0:
                print(loss)

            adjustments = self.do_backward(y)
            self.update_params(adjustments, learning_rate)

    def predict(self, X):
        o = self.do_forward(X)

        return K.argmax(o, axis=1)

    def compile(self, loss=None):
        self.loss_function = losses.get(loss)
        self.is_compiled = True

    def do_forward(self, X):
        o = X
        for layer in self.layers:
            o = layer.forward(o)

        return o

    def do_backward(self, y):
        adjustments = list()

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer is self.layers[-1]:
                delta_y = layer.outputs - y
            else:
                delta_y = K.dot(e, self.layers[i + 1].weight.T)

            e, adjustment = layer.backward(delta_y)
            adjustments.insert(0, adjustment)

        return adjustments

    def update_params(self, adjustments, learning_rate):
        for i, layer in enumerate(self.layers):
            layer.update_params(adjustments[i], learning_rate)