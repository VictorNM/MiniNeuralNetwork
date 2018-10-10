import backend as K


class Model:
    def __init__(self, layers):
        self.layers = layers

    def do_forward(self, x):
        o = x
        for layer in self.layers:
            o = layer.forward(o)

        return o

    def do_backward(self, y):
        adjustments = list()
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                delta_y = layer.outputs - y
            else:
                delta_y = K.dot(e, self.layers[i + 1].weight.T)

            e, delta_weight, delta_bias = layer.backward(delta_y)
            adjustment = {
                'delta_weight': delta_weight,
                'delta_bias': delta_bias
            }
            adjustments.insert(0, adjustment)

        return adjustments

    def update_params(self, adjustments, learning_rate):
        for i, layer in enumerate(self.layers):
            layer.update_params(adjustments[i], learning_rate)

    def fit(self, X, y, n_epochs=2000, learning_rate=0.001):
        for epoch in range(n_epochs):
            self.do_forward(X)
            adjustments = self.do_backward(y)
            self.update_params(adjustments, learning_rate)

    def predict(self, X):
        o = X
        for layer in self.layers:
            o = layer.forward(o)

        return K.argmax(o, axis=1)