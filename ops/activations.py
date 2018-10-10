import backend as K


def sigmoid(x):
    return K.sigmoid(x)


def d_sigmoid(x):
    return K.d_sigmoid(x)


def tanh(x):
    return K.tanh(x)


def d_tanh(x):
    return K.d_tanh(x)


def relu(x):
    return K.relu(x)


def d_relu(x):
    return K.d_relu(x)


def get(id):
    if id is None:
        return None, None

    if id == 'sigmoid':
        return sigmoid, d_sigmoid

    if id == 'tanh':
        return tanh, d_tanh

    if id == 'relu':
        return relu, d_relu