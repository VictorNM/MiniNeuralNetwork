import numpy as np


class Model:
    def __init__(self, layers):
        self.layers = layers

    def do_forward(self, x):
        inputs = x
        delta_weights = list()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer.forward(inputs)

        return inputs

    def do_backward(self, y):
        delta_weights = list()
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                delta_y = y - layer.outputs
            else:
                delta_y = np.dot(e, self.layers[i + 1].weight.T)

            e, delta_weight = layer.backward(delta_y)
            delta_weights.insert(0, delta_weight)

        return delta_weights

    def do_one_epoch(self, x, y, learning_rate):
        outputs = self.do_forward(x)
        delta_weights = self.do_backward(y)
        self.update_params(delta_weights, learning_rate)

    def update_params(self, delta_weights, learning_rate):
        assert len(delta_weights) == len(self.layers)
        for i in range(len(self.layers)):
            self.layers[i].weight += delta_weights[i] * learning_rate

    def fit(self, X, y, n_epochs=2000, learning_rate=0.002):
        for epoch in range(n_epochs):
            self.do_one_epoch(X, y, learning_rate)

    def predict(self, X):
        inputs = X
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer.forward(inputs)

        return np.argmax(inputs, axis=1)