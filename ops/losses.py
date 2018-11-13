import backend as K


def mean_square_error(y, y_hat):
    return K.mean_square_error(y, y_hat)


def mean_square_error_derivative(y, y_hat):
    return K.mean_square_error_derivative(y, y_hat)


def categorical_crossentropy(y, y_hat):
    return K.categorical_crossentropy(y, y_hat)


def categorical_crossentropy_derivative(y, y_hat):
    return K.categorical_crossentropy_derivative(y, y_hat)


def get(id):
    if id == 'mean_squared_error':
        return mean_square_error
    elif id == 'categorical_crossentropy':
        return categorical_crossentropy


def get_derivative(id):
    if id == 'mean_squared_error':
        return mean_square_error_derivative
    elif id == 'categorical_crossentropy':
        return categorical_crossentropy_derivative