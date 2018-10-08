import numpy as np


class NeuralNetwork:
    def __init__(self, model):
        self.model = model

    def do_forward(self, x):
        inputs = x
        delta_weights = list()
        for i in range(len(self.model)):
            layer = self.model[i]
            inputs = layer.forward(inputs)

        return inputs

    def do_backward(self, y):
        delta_weights = list()
        for i in reversed(range(len(self.model))):
            layer = self.model[i]
            if i == len(self.model) - 1:
                delta_y = y - layer.outputs
            else:
                delta_y = np.dot(e, self.model[i+1].weight.T)

            e, delta_weight = layer.backward(delta_y)
            delta_weights.insert(0, delta_weight)

        return delta_weights

    def do_one_epoch(self, x, y):
        outputs = self.do_forward(x)
        delta_weights = self.do_backward(y)
        self.update_params(delta_weights)

    def update_params(self, delta_weights):
        assert len(delta_weights) == len(self.model)
        for i in range(len(self.model)):
            self.model[i].weight += delta_weights[i]

    def fit(self, X, y, n_epochs):
        for epoch in range(n_epochs):
            self.do_one_epoch(X, y)

    def predict(self, X):
        inputs = X
        for i in range(len(self.model)):
            layer = self.model[i]
            inputs = layer.forward(inputs)

        return np.argmax(inputs, axis=1)