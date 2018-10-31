import backend as K


def zeros(shape):
    return K.zeros(shape)


def ones(shape):
    return K.ones(shape)


def random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    return K.random_normal(shape, mean, stddev, seed)


def get(id):
    assert isinstance(id, str)
    assert id in {
        'zeros',
        'ones',
        'random_normal'
    }

    if id == 'zeros':
        return zeros

    if id == 'ones':
        return ones

    if id == 'random_normal':
        return random_normal