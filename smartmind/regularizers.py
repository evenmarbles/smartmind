from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from .utils import to_dtype
from .utils import tolist


def get(name):
    return {
        'l1': uniform,
        'l2': normal,
        'truncated_normal': truncated_normal,
        'constant': constant,
        'xavier_uniform': xavier_uniform,
        'xavier_normal': xavier_normal,
        'lecun_uniform': lecun_uniform,
        'he_uniform': he_uniform,
        'he_normal': he_normal
    }[name]


def get_default(name):
    return {
        'uniform': OrderedDict([('minval', 0.0), ('maxval', 1.0), ('seed', None), ('dtype', 'float32')]),
        'normal': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
        'truncated_normal': OrderedDict([('mean', 0.0), ('stddev', 1.0), ('seed', None), ('dtype', 'float32')]),
        'constant': OrderedDict([('value', 0.0), ('dtype', 'float32')]),
        'xavier_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'xavier_normal': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'lecun_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'he_uniform': OrderedDict([('seed', None), ('dtype', 'float32')]),
        'he_normal': OrderedDict([('seed', None), ('dtype', 'float32')]),
    }[name]


def check_params(name, params):
    if params is None:
        return {}

    default = get_default(name)

    if isinstance(params, dict):
        if not all(k in default for k in params.keys()):
            raise Exception('{}: Initializer parameter mismatch.'.format(name))
        return params

    params = tolist(params)
    if len(params) > len(default):
        raise Exception('{}: Too many initializer parameters given.'.format(name))
    return dict(zip(default.keys(), params))
