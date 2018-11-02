import backend as K


def linear(x):
    return x


def d_linear(x):
    return 1


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
        return linear

    if id == 'sigmoid':
        return sigmoid

    if id == 'tanh':
        return tanh

    if id == 'relu':
        return relu

    raise RuntimeError('Activation is invalid')


def get_derivative(id):
    if id is None:
        return d_linear

    if id == 'sigmoid':
        return d_sigmoid

    if id == 'tanh':
        return d_tanh

    if id == 'relu':
        return d_relu

    raise RuntimeError('Activation is invalid')


def serialize(activation):
    return activation.__name__
