from __future__ import absolute_import

from collections import OrderedDict
from .utils import tolist


class Optimizer(object):
    def __init__(self):
        pass


class SGD(Optimizer):

    def __init__(self):
        super(SGD, self).__init__()


class RMSprop(Optimizer):
    def __init__(self):
        super(RMSprop, self).__init__()


class Adagrad(Optimizer):
    def __init__(self):
        super(Adagrad, self).__init__()


class Adadelta(Optimizer):
    def __init__(self):
        super(Adadelta, self).__init__()


class Adam(Optimizer):
    def __init__(self):
        super(Adam, self).__init__()


class Adamx(Optimizer):
    def __init__(self):
        super(Adamx, self).__init__()


class Nadam(Optimizer):
    def __init__(self):
        super(Nadam, self).__init__()


def get(name):
    return {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamx': Adamx,
        'nadam': Nadam,
    }[name]


def get_default(name):
    return {
        'sgd': OrderedDict([('minval', 0.0), ('maxval', 1.0), ('seed', None), ('dtype', 'float32')]),
        'rmsprop': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
        'adagrad': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
        'adadelta': OrderedDict([('value', 0.0), ('dtype', 'float32')]),
        'adam': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'adamx': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'nadam': OrderedDict([('seed', None), ('dtype', 'float32')]),
    }[name]


def check_params(name, params):
    if params is None:
        return {}

    default = get_default(name)

    if isinstance(params, dict):
        if not all(k in default for k in params.keys()):
            raise Exception('{}: Parameter mismatch.'.format(name))
        return params

    params = tolist(params)
    if len(params) > len(default):
        raise Exception('{}: Too many parameters given.'.format(name))
    return dict(zip(default.keys(), params))
