class Layer:
    def __init__(self):
        self.built = False

    def build(self, inputs_shape):
        self.built = True

    def forward(self, inputs):
        return inputs

    def backward(self, delta_outputs, learning_rate):
        return delta_outputs

    def compute_outputs_shape(self, inputs_shape):
        return inputs_shape
