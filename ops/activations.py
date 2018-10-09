import backend as K


def sigmoid(x):
    return K.sigmoid(x)


def sigmoid_derivative(x):
    return K.sigmoid_derivative(x)


def tanh(x):
    return K.tanh(x)


def tanh_derivative(x):
    return K.tanh_derivative(x)


def relu(x):
    return K.relu(x)


def relu_derivative(x):
    return K.relu_derivative(x)


def get(id):
    if id is None:
        return None, None

    if id == 'sigmoid':
        return sigmoid, sigmoid_derivative

    if id == 'tanh':
        return tanh, tanh_derivative

    if id == 'relu':
        return relu, relu_derivative