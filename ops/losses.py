import backend as K


def mean_square_error(y, y_hat):
    return K.mean_square_error(y, y_hat)


def cross_entropy(y, y_hat):
    return K.cross_entropy(y, y_hat)


def get(id):
    if id == 'mean_squared_error':
        return mean_square_error
    elif id == 'cross_entropy':
        return cross_entropy