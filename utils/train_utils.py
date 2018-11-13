import backend as K


def to_categorical(y, num_classes):
    return K.one_hot(y, num_classes)