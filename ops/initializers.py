import backend as K


def zeros(shape):
    return K.zeros(shape)


def ones(shape):
    return K.ones(shape)


def random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    return K.random_normal(shape, mean, stddev, seed)