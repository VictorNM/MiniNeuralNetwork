import backend as K


def mean_square_diff(y, y_hat):
    return K.mean_square_diff(y, y_hat)


def cross_entropy(y, y_hat):
    return K.cross_entropy(y, y_hat)